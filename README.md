# AI Data Set Generator

프로젝트 **루트 디렉토리**에서 다음 명령으로 웹 화면을 엽니다.

```bash
python3 -B -m asr_corpus.web --host 0.0.0.0 --port 8765
```

같은 컴퓨터에서는 **http://localhost:8765**, 다른 컴퓨터에서는 `http://서버_IP:8765`로 접속합니다.
모델과 실행 도구가 준비된 환경을 기준으로 하며, 기본 경로가 다르면 아래 실행 옵션을 지정합니다.

## 프로젝트 소개

한국어 ASR 학습에 사용할 발화문·정답·WAV 데이터셋을 만드는 웹 도구입니다.
LLM으로 문장을 생성하거나 사용자가 직접 입력하고, 한글 발화문을 TTS로 합성한 뒤
선택한 항목만 학습 데이터셋으로 저장합니다. ASR 모델의 실제 학습은 별도 학습 프로젝트에서 진행합니다.

## 주요 기능

- 금액·금리·온도·날짜·시간·전화번호·소수·수량·순서·비율의 생성 유형과 상대 비율 설정
- 금액의 1~9자리별 생성 비율 지정: 0은 제외, 같은 값은 같은 비율
- 코드가 생성한 숫자와 LLM 템플릿·한글 읽기를 연결해 발화문과 아라비아 숫자 정답 생성
- 생성 단계·템플릿·숫자·배치 결과 확인
- 발화문·정답을 직접 입력하거나 목록의 입력창에서 수정
- 엑셀 업로드: `.xlsx` 첫 시트의 A열 발화문·B열 정답지 추가 (최대 1,000개, 5MB)
- 선택한 발화의 WAV 생성, 행별 재생 및 실제 WAV 파형 탐색
- 체크한 항목 중 WAV가 준비된 데이터만 최종 저장
- 저장된 작업 불러오기, JSONL 내보내기, 생성 데이터 전체 삭제
- 차콜 배경의 다크 테마와 한 화면 작업 배치

## 실행 환경

웹 앱과 숫자 검증 코드는 Python 표준 라이브러리를 사용합니다.
기본 음성 생성은 Apple Silicon의 MLX 환경을 대상으로 합니다.

준비할 항목:

1. Python 3.10 이상
2. 로컬 LLM GGUF 모델과 llama.cpp 실행 파일
3. MOSS-TTS 8B BF16 모델 및 BF16 MOSS Audio Tokenizer 코덱
4. MLX, mlx-audio, NumPy, SoundFile, SciPy가 설치된 TTS 실행용 Python

모델 가중치는 저장소에 포함하지 않으며 추론 중 자동 다운로드하지 않습니다.
기존 환경에서는 다음 경로를 기본값으로 사용합니다.

| 대상 | 기본 경로 |
|---|---|
| LLM | `~/Desktop/personal/dev/ai-models/gemma4-26b-a4b-q4/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` |
| LLM 실행 파일 | `llama-completion` |
| TTS | `~/Desktop/personal/dev/ai-models/MOSS-TTS-8B-BF16` |
| 코덱 | `~/Desktop/personal/dev/ai-models/MOSS-Audio-Tokenizer-BF16` |
| TTS Python | `~/Desktop/personal/dev/ai-models/.voxtral-mlx-tools/bin/python` |

다른 환경에서는 프로젝트 루트에서 경로를 명시합니다.

```bash
python3 -B -m asr_corpus.web \
  --host 0.0.0.0 \
  --port 8765 \
  --model-file /path/to/model.gguf \
  --llama-executable /path/to/llama-completion \
  --tts-model-dir /path/to/MOSS-TTS-8B-BF16 \
  --tts-codec-dir /path/to/MOSS-Audio-Tokenizer-BF16 \
  --tts-python /path/to/tts-environment/bin/python
```

전체 옵션은 `python3 -B -m asr_corpus.web --help`로 확인합니다.
기존 Magpie TTS 백엔드도 `--tts-backend magpie`로 선택할 수 있습니다.

## 사용 순서

1. 생성 프롬프트에 상황·문체·문장 길이를 입력합니다.
2. 숫자 유형과 생성 비율을 정하고 **발화 생성**을 누릅니다.
3. 직접 작성하려면 목록 위의 **직접 추가**에서 발화문과 정답을 같은 줄 수로 입력합니다.
   엑셀은 **엑셀 업로드**에서 `.xlsx`를 선택합니다. 첫 번째 시트의 A열은 발화문, B열은 정답지입니다. 첫 행이 `발화문 / 정답지`이면 제외하고, 제목이 없으면 첫 행부터 읽습니다. 빈 행은 건너뛰며 한쪽 셀이 비면 행 번호를 안내하고 전체 업로드를 취소합니다. 전화번호의 앞자리 0과 날짜 등의 표기를 유지하려면 셀 서식을 텍스트로 지정하세요. 수식은 값으로 붙여넣은 뒤 업로드하세요.
4. 목록에서 문장을 수정하고 사용할 항목을 체크합니다. **채택 항목 전체 선택**으로 확인이 필요 없는 항목을 한 번에 선택할 수 있습니다.
5. **선택한 N개 음성 생성**을 누르거나 행의 음성 생성 버튼을 사용합니다.
6. 재생·파형으로 음성을 확인하고 저장할 항목만 체크합니다.
7. 하단 저장 경로를 정한 뒤 **N개 선택 데이터셋 저장**을 누릅니다.

목록 체크는 음성 생성과 최종 저장에 함께 사용합니다.
체크하지 않은 WAV와 아직 음성이 없는 항목은 최종 데이터셋에 포함하지 않습니다.
문장 수정은 **저장/취소**, **Ctrl·⌘+Enter/Esc**로 처리합니다.

## 모델 메모리 관리

LLM 추론이 끝나면 해당 프로세스를 종료한 뒤 TTS를 실행합니다.
MOSS-TTS 본체로 선택 발화의 음성 코드를 모두 생성한 후 본체를 종료하고,
코덱으로 WAV를 변환한 뒤 코덱 프로세스도 종료합니다.
본체와 코덱을 순차적으로 로딩해 메모리 사용을 줄입니다.

## 저장 위치와 형식

- 작업·설정 기록: `data/asr_web/`
- 기본 WAV·데이터셋 저장 루트: `/tmp/ai-dataset/`
- 저장 루트는 화면 하단 입력창에서 변경할 수 있습니다.
- macOS의 `/tmp`와 `/private/tmp`는 같은 위치를 가리킵니다.

```text
<저장 루트>/
  wav/<작업 ID>/<항목 ID>-<버전>.wav
  dataset-<작업 ID>-<고유 ID>/
    audio/
      <항목 ID>.wav
    transcripts.jsonl
    train.jsonl
    validation.jsonl
    dataset_info.json
```

WAV는 16kHz 모노 PCM16입니다. 전사문 예시:

```json
{"id":"항목 ID","audio":"audio/항목 ID.wav","text":"잔액은 53원입니다.","spoken_text":"잔액은 오십삼 원입니다.","sampling_rate":16000,"num_frames":48000,"duration_seconds":3.0}
```

서로 다른 정답이 2개 이상이면 학습·검증 JSONL도 생성합니다.
같은 정답 그룹이 학습·검증 양쪽에 섞이지 않도록 분리합니다.
외부 학습 코드에서는 상대 오디오 경로와 언어 접두어 처리 방식을 확인해야 합니다.
모델 파일·생성된 WAV·데이터셋·개인 설정은 Git으로 관리하지 않습니다.

## 검증

프로젝트 루트에서 모델 추론 없이 회귀 테스트를 실행합니다.

```bash
python3 -B -m unittest discover -s tests -q
node --check asr_corpus/web_static/app.js
node tests/test_typed_ui.cjs
node tests/test_wav_waveform_ui.cjs
```

Chromium 기반 화면 검증은 기존 실행 파일 경로를 지정합니다.

```bash
CHROME_HEADLESS_SHELL=/path/to/chrome-headless-shell node tests/test_workspace_ui.cjs
CHROME_HEADLESS_SHELL=/path/to/chrome-headless-shell node tests/test_generation_process_ui.cjs
```

`tests/test_asr_audio_ui.cjs`는 별도로 Playwright가 필요합니다.
상세 생성 규칙·API·기존 호환 모드는 [상세 설명](asr_corpus/README.md)을 참고하세요.

현재 알려진 검증 실패: `test_reviewer_cannot_accept_a_pair_that_still_has_missing_month`.
수동 검토 API에 기존 정답을 그대로 재전달하면 수동 수정으로 처리되어,
발화에 월 표현이 누락된 경우에도 채택을 차단하지 못하는 경로가 있습니다.
전체 테스트 실행 시 이 항목은 실패하며, 수동 입력·수정 내용은 사용자가 확인해야 합니다.
