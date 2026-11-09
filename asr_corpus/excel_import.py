"""추가 의존성 없이 XLSX 첫 시트의 발화·정답을 읽는다."""

import base64
import binascii
import posixpath
import re
from io import BytesIO
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree


MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_XML_BYTES = 20 * 1024 * 1024
SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def read_excel_examples(payload):
    """파일 형식과 크기를 확인한 뒤 첫 번째 시트의 입력 쌍을 반환한다."""
    filename = payload.get("filename")
    encoded = payload.get("excel_base64")
    if not isinstance(filename, str) or not filename.lower().endswith(".xlsx"):
        raise ValueError(".xlsx 엑셀 파일을 선택하세요. .xls는 .xlsx로 저장한 뒤 업로드하세요.")
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_FILE_BYTES + 2) // 3):
        raise ValueError("엑셀 파일은 5MB 이하여야 합니다.")
    try:
        contents = base64.b64decode(encoded, validate=True)
        return _read_workbook(contents)
    except (binascii.Error, BadZipFile, ElementTree.ParseError, KeyError, IndexError, TypeError, OSError, RuntimeError, StopIteration) as error:
        raise ValueError("엑셀 파일을 읽을 수 없습니다. 올바른 .xlsx 파일인지 확인하세요.") from error


def _read_workbook(contents):
    """압축 해제 크기를 제한하고 통합 문서에 지정된 첫 시트를 연다."""
    if not contents or len(contents) > MAX_FILE_BYTES:
        raise ValueError("비어 있지 않은 5MB 이하 엑셀 파일을 선택하세요.")
    with ZipFile(BytesIO(contents)) as workbook:
        if len(workbook.infolist()) > 2000 or sum(item.file_size for item in workbook.infolist()) > MAX_XML_BYTES:
            raise ValueError("엑셀 내부 데이터가 너무 큽니다. 필요한 시트와 행만 남겨 업로드하세요.")
        document = _read_xml(workbook, "xl/workbook.xml")
        sheet = document.find(SHEET_NS + "sheets/" + SHEET_NS + "sheet")
        if sheet is None:
            raise ValueError("엑셀 파일에 시트가 없습니다.")
        relationships = _read_xml(workbook, "xl/_rels/workbook.xml.rels")
        relationship = next(item for item in relationships if item.get("Id") == sheet.get(REL_NS + "id"))
        target = relationship.get("Target", "")
        if relationship.get("TargetMode") == "External" or not target:
            raise ValueError("외부 시트는 업로드할 수 없습니다.")
        sheet_path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "xl/" + target)
        shared = _read_shared_strings(workbook)
        return _read_rows(_read_xml(workbook, sheet_path), shared)


def _read_xml(workbook, path):
    """외부 선언을 허용하지 않는 제한된 크기의 XML을 읽는다."""
    data = workbook.read(path)
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML 선언이 포함된 엑셀 파일은 지원하지 않습니다.")
    return ElementTree.fromstring(data)


def _read_shared_strings(workbook):
    """엑셀 공용 문자열과 서식이 나뉜 문자열을 원문으로 복원한다."""
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    return [_text_content(item) for item in _read_xml(workbook, "xl/sharedStrings.xml")]


def _text_content(node):
    """셀의 문자열 조각을 순서대로 이어 붙인다."""
    return "".join(item.text or "" for item in node.iter(SHEET_NS + "t"))


def _cell_text(cell, shared, row_number):
    """문자열과 숫자 셀을 읽고 수식·오류 셀은 행 번호와 함께 거부한다."""
    if cell.find(SHEET_NS + "f") is not None or cell.get("t") in {"e", "b", "d"}:
        raise ValueError(f"엑셀 {row_number}행: 수식·오류·날짜·논리값 대신 텍스트를 입력하세요.")
    if cell.get("t") == "inlineStr":
        return _text_content(cell).strip()
    value = cell.findtext(SHEET_NS + "v", "")
    if cell.get("t") == "s":
        if not value.isdigit():
            raise ValueError(f"엑셀 {row_number}행: 문자열 셀이 올바르지 않습니다.")
        return shared[int(value)].strip()
    return value.strip()


def _row_pair(row, shared, row_number):
    """열 주소로 A·B셀을 찾아 중간 빈 셀 때문에 정답이 이동하지 않게 한다."""
    cells = {}
    for cell in row.findall(SHEET_NS + "c"):
        match = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", cell.get("r", ""))
        if match and match.group(1) in {"A", "B"}:
            cells[match.group(1)] = _cell_text(cell, shared, row_number)
    return cells.get("A", ""), cells.get("B", "")


def _read_rows(sheet, shared):
    """제목과 빈 행을 제외하고 실제 엑셀 행 번호로 입력 오류를 안내한다."""
    examples = []
    for index, row in enumerate(sheet.findall(SHEET_NS + "sheetData/" + SHEET_NS + "row"), 1):
        row_number = row.get("r", str(index))
        spoken, transcript = _row_pair(row, shared, row_number)
        if not spoken and not transcript:
            continue
        if row_number == "1" and (spoken, transcript) == ("발화문", "정답지"):
            continue
        _validate_pair(spoken, transcript, row_number)
        examples.append({"spoken_text": spoken, "normalized_text": transcript})
        if len(examples) > 1000:
            raise ValueError("엑셀 발화는 한 번에 1~1,000개를 추가하세요.")
    if not examples:
        raise ValueError("엑셀 첫 번째 시트의 A열에 발화문, B열에 정답지를 입력하세요.")
    return examples


def _validate_pair(spoken, transcript, row_number):
    """누락·줄바꿈·길이 초과는 저장 전에 원본 행 번호로 알려준다."""
    if not spoken or not transcript:
        raise ValueError(f"엑셀 {row_number}행: A열 발화문과 B열 정답지를 모두 입력하세요.")
    if max(len(spoken), len(transcript)) > 1000 or any(character in spoken + transcript for character in "\r\n"):
        raise ValueError(f"엑셀 {row_number}행: 각 셀은 줄바꿈 없이 1,000자 이하로 입력하세요.")
