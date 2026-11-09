"""엑셀 첫 시트 열 매핑과 저장 전 전체 검증을 확인한다."""

import base64
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from asr_corpus.excel_import import read_excel_examples
from asr_corpus.local import LlamaCppConfig
from asr_corpus.web import DatasetRequestHandler, DatasetWebApplication
from asr_corpus.web_dataset import DatasetJobService


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def excel_payload(rows, shared=None, first_sheet="worksheets/custom.xml"):
    """실제 XLSX 구조의 최소 파일을 메모리에서 생성한다."""
    stream = BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("xl/workbook.xml", f'<workbook xmlns="{MAIN}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="첫 시트" sheetId="7" r:id="first"/><sheet name="둘째" sheetId="1" r:id="second"/></sheets></workbook>')
        workbook.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="first" Target="{first_sheet}"/><Relationship Id="second" Target="worksheets/sheet1.xml"/></Relationships>')
        path = first_sheet.lstrip("/") if first_sheet.startswith("/") else "xl/" + first_sheet
        workbook.writestr(path, f'<worksheet xmlns="{MAIN}"><sheetData>{rows}</sheetData></worksheet>')
        workbook.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{MAIN}"><sheetData/></worksheet>')
        if shared is not None:
            workbook.writestr("xl/sharedStrings.xml", f'<sst xmlns="{MAIN}">{shared}</sst>')
    return {"filename": "발화.XLSX", "excel_base64": base64.b64encode(stream.getvalue()).decode("ascii")}


def row(number, spoken=None, answer=None):
    """셀 주소를 명시해 빈 셀을 생략한 엑셀 행을 만든다."""
    cells = "".join(f'<c r="{column}{number}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
                    for column, text in (("A", spoken), ("B", answer)) if text is not None)
    return f'<row r="{number}">{cells}</row>'


class ExcelImportRegressionTest(unittest.TestCase):
    def test_first_sheet_header_blank_rows_and_extra_columns_are_handled(self):
        """시트 파일명과 무관하게 첫 시트 A·B열을 읽고 제목과 빈 행을 제외한다."""
        rows = row(1, "발화문", "정답지") + row(2) + row(4, "  잔액은 오십삼 원입니다.  ", "잔액은 53원입니다.")
        rows += '<row r="5"><c r="C5" t="inlineStr"><is><t>무시</t></is></c></row>'
        payload = excel_payload(rows, first_sheet="/xl/worksheets/custom.xml")
        self.assertEqual(read_excel_examples(payload), [{"spoken_text": "잔액은 오십삼 원입니다.", "normalized_text": "잔액은 53원입니다."}])

    def test_headerless_shared_rich_text_and_numeric_zero_are_preserved(self):
        """제목 없는 첫 행·공용 서식 문자열·숫자 0·전화번호 앞자리 0을 유지한다."""
        rows = '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>0</v></c></row>' + row(2, "공일공", "010")
        examples = read_excel_examples(excel_payload(rows, '<si><r><t>영</t></r><r><t>입니다.</t></r></si>'))
        self.assertEqual(examples, [{"spoken_text": "영입니다.", "normalized_text": "0"}, {"spoken_text": "공일공", "normalized_text": "010"}])

    def test_missing_cell_reports_original_row_number(self):
        """빈 셀을 다음 열로 당기지 않고 실제 행 번호로 오류를 안내한다."""
        for spoken, answer in ((None, "정답"), ("발화", None)):
            with self.subTest(spoken=spoken), self.assertRaisesRegex(ValueError, "엑셀 8행"):
                read_excel_examples(excel_payload(row(1, "발화문", "정답지") + row(8, spoken, answer)))

    def test_empty_formula_error_multiline_and_oversized_batches_are_rejected(self):
        """빈 시트·제목만 있는 파일·수식·오류·줄바꿈·개수 초과를 거부한다."""
        invalid = ["", row(1, "발화문", "정답지"), row(3, "줄\n바꿈", "정답"), row(3, "가" * 1001, "정답"),
                   '<row r="3"><c r="A3"><f>1+1</f><v>2</v></c><c r="B3"><v>2</v></c></row>',
                   '<row r="3"><c r="A3" t="e"><v>#REF!</v></c><c r="B3"><v>2</v></c></row>',
                   "".join(row(index, str(index), str(index)) for index in range(1, 1002))]
        for rows in invalid:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                read_excel_examples(excel_payload(rows))
        self.assertEqual(len(read_excel_examples(excel_payload("".join(row(index, str(index), str(index)) for index in range(1, 1001))))), 1000)

    def test_malformed_and_unsupported_files_are_rejected(self):
        """잘못된 파일·인코딩·내부 XML·공용 문자열 참조를 오류로 반환한다."""
        invalid = [{"filename": "test.xls", "excel_base64": ""}, {"filename": "test.xlsx", "excel_base64": "!"},
                   {"filename": "test.xlsx", "excel_base64": base64.b64encode(b"invalid zip").decode()},
                   excel_payload('<row r="1"><c r="A1" t="s"><v>99</v></c></row>'),
                   excel_payload('<row>')]
        for payload in invalid:
            with self.subTest(filename=payload["filename"]), self.assertRaises(ValueError):
                read_excel_examples(payload)

    def test_large_excel_request_is_allowed_only_on_manual_addition_routes(self):
        """큰 업로드는 수동 추가 경로만 허용하고 다른 API의 크기 제한을 유지한다."""
        from email.message import Message
        handler = object.__new__(DatasetRequestHandler)
        contents = json.dumps({"filename": "test.xlsx", "excel_base64": "a" * 70000}).encode()
        handler.headers = Message()
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = str(len(contents))
        for path in ("/api/jobs/manual", "/api/jobs/" + "a" * 32 + "/examples"):
            handler.path = path
            handler.rfile = BytesIO(contents)
            self.assertEqual(handler._read_payload()["filename"], "test.xlsx")
        handler.path = "/api/jobs"
        with self.assertRaises(ValueError):
            handler._read_payload()
        handler.path = "/api/jobs/manual"
        handler.headers.replace_header("Content-Length", str(8 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            handler._read_payload()

    def test_import_create_append_and_failure_preserve_saved_rows_without_model(self):
        """모델 없이 새 작업과 기존 목록에 추가하고 오류·중복 시 기존 파일을 보존한다."""
        with tempfile.TemporaryDirectory(prefix="excel-import-test-") as directory:
            root = Path(directory)
            factory = Mock(side_effect=AssertionError("모델 실행 금지"))
            model = root / "test.gguf"
            model.write_bytes(b"GGUFtest")
            config = LlamaCppConfig(model, executable=sys.executable)
            model.unlink()
            service = DatasetJobService(config, root / "jobs", factory)
            self.addCleanup(service.close)
            application = DatasetWebApplication(service)
            payload = {**excel_payload(row(1, "오십삼 원", "53원")), "output_root": str(root / "output")}
            response = application.dispatch("POST", "/api/jobs/manual", payload)
            self.assertEqual(response.status, 201, response.body)
            job = json.loads(response.body)
            endpoint = f"/api/jobs/{job['id']}/examples"
            response = application.dispatch("POST", endpoint, excel_payload(row(1, "사과 두 개", "사과 2개")))
            self.assertEqual(response.status, 201, response.body)
            updated = json.loads(response.body)
            self.assertEqual(updated["examples"][0], job["examples"][0])
            self.assertEqual(updated["counts"]["accepted"], 2)
            saved = root / "jobs" / job["id"] / "job.json"
            before = saved.read_bytes()
            for invalid in (row(1, "새 문장", "새 문장") + row(7, "누락", None), row(1, "오십삼 원", "다른 정답")):
                response = application.dispatch("POST", endpoint, excel_payload(invalid))
                self.assertEqual(response.status, 400, response.body)
                self.assertEqual(saved.read_bytes(), before)
            service._jobs[job["id"]]["status"] = "running"
            response = application.dispatch("POST", endpoint, excel_payload(row(1, "새 문장", "새 문장")))
            self.assertEqual(response.status, 409)
            service._jobs[job["id"]]["status"] = "completed"
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
