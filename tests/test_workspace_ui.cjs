"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");

// 실제 HTML·CSS·JS와 모의 API만으로 단일 작업 화면의 전체 조작을 검증한다.
async function main() {
  const browser = process.env.CHROME_HEADLESS_SHELL;
  assert.ok(browser, "CHROME_HEADLESS_SHELL을 지정하세요.");
  const root = path.join(__dirname, "../asr_corpus/web_static");
  const css = fs.readFileSync(path.join(root, "styles.css"), "utf8");
  const script = fs.readFileSync(path.join(root, "app.js"), "utf8").replace(/initialize\(\);\s*$/, "");
  const wav = Buffer.alloc(44 + 32000);
  wav.write("RIFF", 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write("data", 36); wav.writeUInt32LE(32000, 40);
  for (let index = 0; index < 16000; index++) wav.writeInt16LE(Math.round(Math.sin(index / 8) * (index < 8000 ? 4000 : 22000)), 44 + index * 2);
  const test = `
    function check(value, message) { if (!value) throw new Error(message); }
    function snapshot() { return JSON.parse(JSON.stringify(job)); }
    const requests = [];
    let deleted = false;
    let manualSnapshot = null;
    let manualFail = false;
    const audioData = ${JSON.stringify("data:audio/wav;base64," + wav.toString("base64"))};
    const definitions = [
      ["money", "결제 예정 금액은 삼만 오천 원입니다.", "결제 예정 금액은 35,000원입니다.", "결제 예정 금액은 {number}입니다.", "35,000원", "삼만 오천 원"],
      ["temperature", "오늘의 최저 기온은 영하 삼점오 도씨입니다.", "오늘의 최저 기온은 -3.5℃입니다.", "오늘의 최저 기온은 {number}입니다.", "-3.5℃", "영하 삼점오 도씨"],
      ["date", "예약 날짜는 십일월 십오일입니다.", "예약 날짜는 11월 15일입니다.", "예약 날짜는 {number}입니다.", "11월 15일", "십일월 십오일"],
      ["count", "추가 주문은 커피 다섯 잔입니다.", "추가 주문은 커피 5잔입니다.", "추가 주문은 커피 {number}입니다.", "5잔", "다섯 잔"],
      ["time", "다음 회의는 오후 두 시입니다.", "다음 회의는 오후 2시입니다.", "다음 회의는 오후 {number}입니다.", "2시", "두 시"],
      ["phone", "연락처 확인이 필요합니다.", "연락처를 확인해 주세요.", "연락처는 {number}입니다.", "010-1234-5678", "공일공 일이삼사 오육칠팔"]
    ];
    const rows = definitions.map(([type, spoken, normalized, template, value, reading], index) => ({
      id: String(index + 1).repeat(32), type, spoken_text: spoken, normalized_text: normalized,
      template, generated_number: {type, value}, protected_terms: [], accepted: index <= 4,
      needs_review: index === 4, reason: index === 4 ? "near_duplicate" : index === 5 ? "invalid" : "unique",
      quality_issue: index === 5 ? "전화번호와 한글 읽기 대응을 확인하세요." : null,
      audio: index < 3 ? {status: "ready", fingerprint: String(index + 1).repeat(64), duration_seconds: 2.1, included: true, url: audioData, peak_memory_gib: 15.9} : {status: "missing"},
      candidate: index === 5 ? {number_generation_issue: "전화번호 형식 확인"} : undefined
    }));
    const job = {id: "a".repeat(32), title: "고객 상담 · 숫자 발화", status: "completed", message: "발화 생성이 완료되었습니다.",
      settings: {prompt: "고객 상담에 필요한 숫자 발화를 생성해줘.", count: 6, batch_size: 3, output_root: "/tmp/workspace-data", auto_audio: false, type_weights: {money: 3, temperature: 2, date: 1}},
      counts: {accepted: 4}, examples: rows, model_runtime: {active_model: null, last_released: "tts"},
      audio: {status: "completed", message: "음성 생성 완료", counts: {total: 4, ready: 3, failed: 0}},
      generation_batches: [{index: 1, phase: "completed", status: "completed", examples: rows.slice(0, 3).map((row, index) => ({id: String(index), result_id: row.id, type: row.type, template: row.template, value: row.generated_number.value, spoken: definitions[index][5], spoken_text: row.spoken_text, normalized_text: row.normalized_text, accepted: row.accepted}))}]};
    api = async (url, options = {}) => {
      const payload = options.body ? JSON.parse(options.body) : null;
      requests.push({url, method: options.method || "GET", payload});
      if ((url.endsWith("/examples") || url === "/api/jobs/manual") && options.method === "POST") {
        if (manualFail) throw new Error("수동 저장 테스트 오류");
        const inputExamples = payload.excel_base64 ? [{spoken_text: "엑셀 발화", normalized_text: "엑셀 정답"}] : payload.examples;
        const added = inputExamples.map((row, index) => ({...row, id: payload.excel_base64 ? "e".repeat(32) : "b".repeat(30) + String(index + 10).padStart(2, "0"), source: "manual", accepted: true, needs_review: false, quality_issue: null, protected_terms: [], manually_edited: true, reason: "manual_review", audio: {status: "missing"}}));
        if (url === "/api/jobs/manual") {
          manualSnapshot = {...snapshot(), id: "f".repeat(32), title: "직접 입력 발화", examples: added, generation_batches: [], counts: {accepted: added.length}, settings: {...job.settings, prompt: "직접 입력 발화", count: added.length, type_weights: null}};
          return JSON.parse(JSON.stringify(manualSnapshot));
        }
        job.examples.push(...added);
        job.counts.accepted += added.length;
        return snapshot();
      }
      if (url === "/api/jobs" && options.method === "POST") { job.settings = {...job.settings, ...payload}; return snapshot(); }
      if (url === "/api/data/deletion-preview") return {token: "preview-token", jobs: 1, examples: 6, wav_files: 4, datasets: 1, files: 12, roots: ["/tmp/workspace-data"], skipped: 0};
      if (url === "/api/data/delete-all") { check(payload.confirmed && payload.token === "preview-token", "확인한 삭제 토큰 전달"); deleted = true; return {deleted_files: 12, deleted_wav_files: 4}; }
      if (url === "/api/jobs") return {jobs: deleted ? [] : [...(manualSnapshot ? [{id: manualSnapshot.id, title: manualSnapshot.title, status: manualSnapshot.status}] : []), {id: job.id, title: job.title, status: job.status}]};
      if (url.endsWith("/mapping")) {
        const result = {count: payload.example_ids.length, output_dir: payload.output_root + "/dataset-workspace", output_root: payload.output_root};
        job.audio.last_mapping = result;
        return result;
      }
      if (url.endsWith("/audio-review")) {
        const id = url.split("/").at(-2);
        job.examples.find((row) => row.id === id).audio.included = payload.included;
      } else if (url.endsWith("/audio") && options.method === "POST") {
        for (const row of job.examples.filter((item) => payload.example_ids.includes(item.id))) row.audio = {...row.audio, status: "ready", url: audioData, duration_seconds: 2.1, fingerprint: row.id + row.id, included: true};
      } else if (options.method === "PATCH" && url.includes("/examples/")) {
        const row = job.examples.find((item) => item.id === url.split("/").at(-1));
        Object.assign(row, payload, {needs_review: false, quality_issue: null, reason: "manual_review"});
      }
      return snapshot();
    };
    state.connected = true;
    state.audioAvailable = true;
    restoreMoneyDigitWeights({"2": 1, "9": 3});
    bindEvents();
    render();
    check(!document.querySelector("#money-digit-settings").hidden, "금액 선택 시 자릿수 설정 표시");
    const moneyChoice = document.querySelector('[data-number-type="money"]');
    moneyChoice.click();
    check(document.querySelector("#money-digit-settings").hidden && document.querySelector("[data-money-digit]").disabled, "금액 해제 시 설정 숨김·비활성화");
    moneyChoice.click();
    check(!document.querySelector("#money-digit-settings").hidden && !document.querySelector("[data-money-digit]").disabled, "금액 재선택 시 설정 복원");
    check(!document.querySelector("header, #audio-stage, [data-stage]"), "헤더·상세 패널·탭 제거");
    check(!document.querySelector('#search, input[type="search"]'), "검색창 제거");
    check(!document.querySelector(".workspace-heading").textContent.includes("DATA WORKSPACE"), "작업공간 영문 제목 제거");
    document.querySelector("#prompt").value = job.settings.prompt;
    document.querySelector("#count").value = "6";
    restoreTypeWeights(job.settings.type_weights);
    syncRoot("/tmp/workspace-data");
    await startGeneration({preventDefault() {}});
    check(requests[0].payload.count === 6 && requests[0].payload.auto_audio === false, "발화만 먼저 생성");
    check(JSON.stringify(requests[0].payload.money_digit_weights) === JSON.stringify({"2": 1, "9": 3}), "자릿수 비율을 생성 요청에 전달");
    check(document.querySelectorAll("#text-list .utterance-row").length === 6, "전체 발화 목록");
    check(generationRows().length === 0 && document.querySelector("#tts-generate").disabled, "명시적으로 선택하기 전 음성 생성 비활성화");
    check(document.querySelector("#create-mapping").disabled && includedRows().length === 0, "WAV가 있어도 미선택 시 저장 비활성화");
    const requestCount = requests.length;
    await createMapping();
    check(requests.length === requestCount, "미선택 저장 요청 차단");
    const filter = document.querySelector("#list-filter");
    filter.value = "accepted";
    filter.dispatchEvent(new Event("change", {bubbles: true}));
    check(document.querySelectorAll(".utterance-row").length === 4 && generationRows().length === 0, "체크 여부와 독립적으로 확인 불필요 채택 항목만 표시");
    filter.value = "selected";
    filter.dispatchEvent(new Event("change", {bubbles: true}));
    check(document.querySelectorAll(".utterance-row").length === 0, "생성 선택 필터와 채택 필터 구분");
    filter.value = "all";
    filter.dispatchEvent(new Event("change", {bubbles: true}));
    check(document.querySelectorAll("#text-list .inline-player").length === 3, "완료된 행마다 재생기 표시");
    check(!document.querySelector("#generation-process").open, "완료 배치 기록 접기");
    const panes = ["setup-stage", "text-stage"].map((id) => document.getElementById(id).getBoundingClientRect());
    if (window.innerWidth > 760) {
      check(Math.abs(panes[0].top - panes[1].top) < 2 && panes[0].right < panes[1].left, "두 열 배치");
      check(document.documentElement.scrollHeight <= window.innerHeight + 2, "데스크톱 전체 화면 세로 넘침 없음");
    } else {
      check(document.querySelector("#setup-stage").classList.contains("is-collapsed"), "모바일 설정 접기");
      document.querySelector("#toggle-settings").click();
      check(document.querySelector("#toggle-settings").getAttribute("aria-expanded") === "true", "설정 펼치기");
      document.querySelector("#toggle-settings").click();
    }
    const wave = document.querySelector(".waveform");
    await wave.loadWaveform();
    // 관찰자가 먼저 요청했더라도 실제 WAV 계산 완료까지 기다린다.
    const parsed = await loadWaveform(document.querySelector("audio").src);
    await new Promise((resolve) => setTimeout(resolve, 0));
    check(parsed.peaks.length === 96 && parsed.duration === 1, "실제 PCM 구간·길이 계산");
    check(parsed.peaks[5] < parsed.peaks[90], "샘플 진폭 변화가 파형에 반영");
    check(wave.querySelector("path") && wave.dataset.loaded === "true", "실제 SVG 파형 표시");
    const player = document.querySelector("audio");
    const playing = new Promise((resolve) => player.addEventListener("playing", resolve, {once: true}));
    document.querySelector(".play-button").click();
    await playing;
    check(!player.paused, "재생 버튼으로 실제 WAV 재생");
    document.querySelector("[data-text-choice]").click();
    check(generationRows().length === 1, "체크한 발화만 생성 선택");
    setJob(snapshot());
    check(document.querySelector("audio") === player && !player.paused, "선택·조회 중 재생 유지");
    wave.dispatchEvent(new KeyboardEvent("keydown", {key: "End", bubbles: true}));
    check(player.currentTime > 0.9, "파형 키보드 탐색");
    player.pause();
    check(!document.querySelector("[data-audio-choice], .wav-download"), "저장 포함·개별 WAV 다운로드 제거");
    job.examples[0].audio.included = false;
    setJob(snapshot());
    check(document.querySelector("audio") === player, "조회 중 재생기 유지");
    document.querySelector('[data-text-choice="' + rows[3].id + '"]').click();
    check(generationRows().length === 2 && includedRows().length === 1, "선택한 미생성 WAV는 저장에서 제외");
    await createMapping();
    const mapping = requests.findLast((request) => request.url.endsWith("/mapping"));
    check(mapping.payload.example_ids.length === 1 && mapping.payload.example_ids[0] === rows[0].id && mapping.payload.include_all_ready === true, "이전 개별 제외 기록과 관계없이 현재 체크한 준비 WAV만 저장");
    check(document.querySelector("#mapping-path").textContent.includes("dataset-workspace"), "저장 결과 표시");
    document.querySelector('[data-text-choice="' + rows[3].id + '"]').click();
    await startAudio();
    check(requests.findLast((request) => request.url.endsWith("/audio")).payload.example_ids.join() === rows[0].id, "체크한 항목만 음성 생성");
    const fourth = document.querySelector('[data-row-id="' + rows[3].id + '"]');
    check(!fourth.querySelector("audio"), "미생성 행에는 재생기 없음");
    fourth.querySelector("[data-row-generate]").click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    check(requests.findLast((request) => request.url.endsWith("/audio")).payload.example_ids.join() === rows[3].id, "행 버튼에서 개별 생성");
    check(document.querySelector('[data-row-id="' + rows[3].id + '"] audio'), "생성 완료 후 재생기 추가");
    document.querySelector('[data-text-choice="' + rows[0].id + '"]').click();
    document.querySelector('[data-text-choice="' + rows[1].id + '"]').click();
    check(document.querySelector("#create-mapping").textContent.includes("1개 선택"), "저장 버튼에 실제 선택 WAV 개수 표시");
    await createMapping();
    const changedMapping = requests.findLast((request) => request.url.endsWith("/mapping"));
    check(changedMapping.payload.example_ids.join() === rows[1].id, "체크 해제한 WAV는 저장에서 제외");
    check(!document.querySelector(".row-editor"), "별도 수정 펼치기 제거");
    const editor = document.querySelectorAll(".inline-text-editor")[1];
    const input = editor.querySelector('[data-edit-field="spoken"]');
    check(input.tagName === "INPUT" && input.type === "text" && !input.hidden && !editor.querySelector("textarea, [data-edit-text]"), "처음부터 고정된 입력창 표시");
    input.focus();
    input.click();
    check(editor.querySelector('[data-edit-field="spoken"]') === input, "클릭해도 입력창 교체 없음");
    input.value = "오늘의 기온은 영하 삼점오 도씨입니다.";
    input.dispatchEvent(new Event("input", {bubbles: true}));
    setJob(snapshot());
    check(document.querySelectorAll(".inline-text-editor")[1] === editor && input.value.includes("오늘의 기온"), "조회 중 수정 초안 보존");
    editor.querySelector("[data-row-save]").click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    check(job.examples[1].spoken_text === input.value, "목록에서 수정 저장");
    const savedEditor = document.querySelectorAll(".inline-text-editor")[1];
    check(!savedEditor.querySelector('[data-edit-field="spoken"]').hidden && savedEditor.querySelector(".inline-edit-actions").hidden, "저장 후에도 입력창 유지");
    const originalAnswer = job.examples[1].normalized_text;
    const answerInput = savedEditor.querySelector('[data-edit-field="normalized"]');
    check(answerInput.tagName === "INPUT" && !answerInput.hidden, "정답도 고정 입력창");
    answerInput.value = "취소할 정답";
    answerInput.dispatchEvent(new Event("input", {bubbles: true}));
    answerInput.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape", bubbles: true}));
    check(!answerInput.hidden && answerInput.value === originalAnswer && !state.rowDrafts[rows[1].id], "Esc로 초안 취소·원문 복구");
    answerInput.value = "오늘의 최저 기온은 -3.5℃입니다. 수정";
    answerInput.dispatchEvent(new Event("input", {bubbles: true}));
    answerInput.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", ctrlKey: true, bubbles: true}));
    await new Promise((resolve) => setTimeout(resolve, 0));
    check(job.examples[1].normalized_text === answerInput.value, "Ctrl+Enter로 정답 저장");
    await bulkText(true);
    await startAudio();
    check(requests.findLast((request) => request.url.endsWith("/audio")).payload.example_ids.length === 4, "생성 가능 항목만 전체 선택");
    await bulkText(false);
    check(!generationRows().length && selectedRows().length === 4, "선택 해제는 채택 상태 보존");
    document.querySelector("#select-valid").click();
    check(generationRows().length === 4 && document.querySelectorAll("[data-text-choice]:checked").length === 4, "채택 항목 전체 선택 버튼 동작");
    check(!state.generationSelection.has(rows[4].id) && !state.generationSelection.has(rows[5].id), "확인 필요·품질 오류 항목은 전체 선택에서 제외");
    check(document.querySelector("#select-valid").disabled, "채택 전체 선택 완료 후 중복 선택 비활성화");
    state.filter = "review"; render();
    check(document.querySelectorAll(".utterance-row").length === 1, "확인 필요 필터");
    state.filter = "all"; job.status = "running"; setJob(snapshot());
    check(document.querySelector("[data-money-digit]").disabled, "생성 중 자릿수 설정 보호");
    check(document.querySelector("#prompt").disabled && document.querySelector("[data-row-edit]").disabled && document.querySelector("#create-mapping").disabled, "진행 중 수정·저장 보호");
    job.status = "completed"; setJob(snapshot());
    document.querySelector("#add-manual").click();
    check(!document.querySelector("#manual-form").hidden && document.activeElement === document.querySelector("#manual-spoken"), "직접 추가 버튼으로 인라인 입력 폼 열기");
    const beforeManual = job.examples.length;
    const beforeManualRequests = requests.length;
    document.querySelector("#manual-spoken").value = ["잔액은 오십삼 원입니다.", "오늘은 영하 삼 점 오 도입니다."].join(String.fromCharCode(10));
    document.querySelector("#manual-transcript").value = "잔액은 53원입니다.";
    await submitManualForm({preventDefault() {}});
    check(requests.length === beforeManualRequests && !document.querySelector("#manual-alert").hidden, "줄 수 불일치는 저장 요청 전에 차단");
    document.querySelector("#manual-transcript").value = ["잔액은 53원입니다.", "오늘은 -3.5℃입니다."].join(String.fromCharCode(10));
    const draftSpoken = document.querySelector("#manual-spoken").value;
    manualFail = true;
    await submitManualForm({preventDefault() {}});
    check(!document.querySelector("#manual-form").hidden && document.querySelector("#manual-spoken").value === draftSpoken && job.examples.length === beforeManual, "저장 오류 시 입력 내용과 기존 목록 유지");
    manualFail = false;
    const preservedPlayer = document.querySelector("audio");
    await submitManualForm({preventDefault() {}});
    const manualRequest = requests.findLast((request) => request.url.endsWith("/examples"));
    check(manualRequest.payload.examples.length === 2 && manualRequest.payload.examples[1].normalized_text === "오늘은 -3.5℃입니다.", "발화와 정답을 줄별로 연결");
    check(job.examples.length === beforeManual + 2 && document.querySelector("#manual-form").hidden, "기존 목록에 직접 입력 2개 추가");
    check(job.examples.slice(-2).every((row) => state.generationSelection.has(row.id)), "새 항목을 생성·저장 대상으로 자동 선택");
    check(document.querySelector("audio") === preservedPlayer, "직접 추가 중 기존 재생기 유지");
    check(document.querySelector('[data-row-id="' + job.examples.at(-1).id + '"]').textContent.includes("직접 입력"), "직접 입력 출처 표시");
    document.querySelector("#new-job").click();
    document.querySelector("#add-manual").click();
    check(document.querySelector("#manual-target").textContent === "새 작업으로 저장", "새 목록에서 직접 입력 가능");
    document.querySelector("#manual-spoken").value = "다음 역입니다.";
    document.querySelector("#manual-transcript").value = "다음 역입니다.";
    await submitManualForm({preventDefault() {}});
    check(state.job.id === "f".repeat(32) && state.job.examples.length === 1 && generationRows().length === 1, "LLM 생성 없이 새 직접 입력 작업 표시");
    check(requests.some((request) => request.url === "/api/jobs/manual"), "직접 입력 전용 새 작업 요청");
    check(document.querySelector("#delete-data").textContent === "전체 삭제", "삭제 버튼 압축");
    check(document.querySelector("#delete-dialog").textContent.includes("생성 WAV"), "삭제 대상 WAV 안내");
    check(document.documentElement.scrollWidth <= window.innerWidth, "가로 넘침 없음");
    await previewDeletion();
    check(document.querySelector("#delete-dialog").open && document.querySelector("#delete-summary").textContent.includes("WAV 파일 4개"), "삭제 전 WAV 개수 표시");
    document.querySelector("#cancel-delete").click();
    check(!requests.some((request) => request.url.endsWith("delete-all")), "취소 시 파일 삭제 안함");
    await previewDeletion();
    await deleteAllData();
    check(!state.job && !document.querySelector("audio") && !document.querySelector(".utterance-row"), "삭제 후 목록·재생기 정리");
    check(!document.querySelector("#delete-dialog").open && document.querySelector("#save-result").hidden, "삭제 창·저장 결과 정리");
    // 목소리 지정 UI 제거와 유지된 엑셀 업로드를 검증한다.
    setJob(snapshot());
    check(!document.querySelector("#voice-select, #apply-voice, #upload-voice, #voice-file, [data-row-voice]"), "목소리 지정·등록 UI 제거");
    const excelFile = new File([new Uint8Array([80, 75, 3, 4])], "examples.xlsx");
    await uploadExcel({target: {files: [excelFile], value: "examples.xlsx"}});
    check(requests.findLast((request) => request.url.endsWith("/examples")).payload.excel_base64 === "UEsDBA==", "실제 FileReader로 엑셀 업로드 유지");
    await startAudio(["e".repeat(32)]);
    const excelAudio = requests.findLast((request) => request.url.endsWith("/audio") && request.method === "POST").payload;
    check(!("voice_id" in excelAudio) && !("reference_example_id" in excelAudio), "엑셀 발화는 목소리 지정 없이 생성");
    // 최종 배치 스크린샷은 생성 결과가 있는 화면으로 남긴다.
    setJob(snapshot());
    openManualForm();
    document.querySelector("#manual-spoken").value = "잔액은 오십삼 원입니다.";
    document.querySelector("#manual-transcript").value = "잔액은 53원입니다.";
    check(document.documentElement.scrollWidth <= window.innerWidth, "직접 입력 폼의 가로 넘침 없음");
    if (window.innerWidth > 760) check(document.documentElement.scrollHeight <= window.innerHeight + 2, "직접 입력 중에도 데스크톱 한 화면 유지");
    document.querySelector("#toast").hidden = true;
    document.body.dataset.testResult = "passed";
  `;
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "asr-workspace-ui-"));
  try {
    let html = fs.readFileSync(path.join(root, "index.html"), "utf8");
    html = html.replace('<link rel="stylesheet" href="/styles.css"><script src="/app.js" defer></script>', "<style>" + css + "</style>");
    html = html.replace("</body>", "<script>" + script + "\n(async () => {try {" + test + "} catch (error) {document.body.dataset.testResult='failed';document.body.dataset.testError=error.message;}})();<" + "/script></body>");
    const fixture = path.join(directory, "fixture.html");
    fs.writeFileSync(fixture, html);
    for (const [width, height] of [[1440, 900], [1100, 800], [900, 900], [390, 1000]]) {
      const args = ["--no-sandbox", "--disable-gpu", "--disable-background-networking", "--no-first-run", "--hide-scrollbars", "--mute-audio", "--autoplay-policy=no-user-gesture-required", "--window-size=" + width + "," + height, "--remote-debugging-pipe"];
      const child = spawn(browser, args, {stdio: ["ignore", "ignore", "ignore", "pipe", "pipe"]});
      let sequence = 0;
      let buffer = "";
      const pending = new Map();
      child.stdio[4].on("data", (data) => {
        buffer += data.toString();
        let boundary;
        while ((boundary = buffer.indexOf("\0")) >= 0) {
          const message = JSON.parse(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 1);
          if (pending.has(message.id)) { pending.get(message.id)(message); pending.delete(message.id); }
        }
      });
      const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
        const id = ++sequence;
        const timer = setTimeout(() => { pending.delete(id); reject(new Error("브라우저 검증 시간 초과: " + method)); }, 15000);
        pending.set(id, (message) => { clearTimeout(timer); if (message.error) reject(new Error(message.error.message)); else resolve(message.result); });
        child.stdio[3].write(JSON.stringify({id, method, params, sessionId}) + "\0");
      });
      let output;
      try {
        const {targetId} = await call("Target.createTarget", {url: "about:blank"});
        const {sessionId} = await call("Target.attachToTarget", {targetId, flatten: true});
        await call("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: false}, sessionId);
        await call("Page.navigate", {url: "file://" + fixture}, sessionId);
        await new Promise((resolve) => setTimeout(resolve, 100));
        await call("Runtime.evaluate", {expression: "(async () => {for(let i=0;i<100 && !document.body?.dataset.testResult;i++) await new Promise(r=>setTimeout(r,100));})()", awaitPromise: true}, sessionId);
        const result = await call("Runtime.evaluate", {expression: "document.documentElement.outerHTML", returnByValue: true}, sessionId);
        output = result.result.value;
        if (process.env.WORKSPACE_SCREENSHOT_DIR) {
          const screenshot = await call("Page.captureScreenshot", {format: "png"}, sessionId);
          fs.writeFileSync(path.join(process.env.WORKSPACE_SCREENSHOT_DIR, "workspace-" + width + ".png"), Buffer.from(screenshot.data, "base64"));
        }
      } finally { child.kill(); }
      assert.ok(/data-test-result="passed"/.test(output), output.match(/data-test-error="([^"]+)"/)?.[1] || "브라우저 검증 결과 없음");
      console.log(width + "px 단일 화면·목소리 지정 제거·엑셀 업로드·생성·저장 검증 통과");
    }
  } finally { fs.rmSync(directory, {recursive: true, force: true}); }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
