"use strict";

const EXAMPLES = {
  temperature: "일기예보와 냉장 보관에 관한 자연스러운 한국어 발화를 만들어줘. 영하·영상과 소수점 온도를 다양하게 포함해줘. 정답은 -3.5℃처럼, 발화는 영하 삼 점 오 도처럼 표현해줘.",
  support: "배송 문의, 예약 변경, 환불 안내에 관한 한국어 고객 상담 발화를 다양하게 만들어줘. 필요한 경우 금액·인원·날짜를 포함하고 정답에는 아라비아 숫자를 써줘.",
  terms: "정보 기술 분야의 약어와 전문 용어를 포함한 자연스러운 한국어 발화를 만들어줘. 발화에는 한국어로 읽을 발음을, 정답에는 요청한 약어 표기를 유지해줘."
};
const ACTIVE = new Set(["queued", "running", "cancelling"]);
const STATUS_LABELS = { queued: "준비 중", running: "생성 중", cancelling: "중지 중", completed: "생성 완료", partial: "일부 생성", failed: "실행 실패", cancelled: "중지됨", interrupted: "중단됨" };
const state = { job: null, generationSelection: new Set(), rowDrafts: {}, audioRequestedIds: [], filter: "all", connected: false, busy: false, timer: null, editingId: null, manualJobId: null, audioAvailable: false, audioError: "", preferences: null, preferencesLoaded: false, saveResult: null, deletionPreview: null, deleting: false };
const $ = (selector) => document.querySelector(selector);
let toastTimer;
let waveformObserver = null;
const waveformCache = new Map();
const waveformControllers = new Map();

// 사용자 문자열을 HTML로 해석하지 않고 요소를 만든다.
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// 화면에서 정의한 아이콘만 사용한다.
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon");
  const use = document.createElementNS(svg.namespaceURI, "use");
  use.setAttribute("href", "#i-" + name);
  svg.append(use);
  return svg;
}

// 로컬 앱의 JSON 응답과 오류를 공통으로 처리한다.
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...options.headers } });
  } catch (_) { throw new Error("로컬 앱에 연결할 수 없습니다. 실행 상태를 확인해 주세요."); }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "요청을 처리하지 못했습니다.");
  return data;
}

// 입력 초안과 작업 단계만 브라우저에 저장한다.
function remember(key, value) {
  try { localStorage.setItem("asr-studio-" + key, value); } catch (_) { /* 입력은 화면에 유지한다. */ }
}

// 브라우저 저장을 지원하지 않아도 기본 흐름을 유지한다.
function recalled(key) {
  try { return localStorage.getItem("asr-studio-" + key); } catch (_) { return null; }
}

// 선택한 숫자 유형과 상대 비율을 요청 설정으로 만든다.
function selectedTypeWeights() {
  const weights = {};
  for (const checkbox of document.querySelectorAll("[data-number-type]")) {
    if (checkbox.checked) weights[checkbox.dataset.numberType] = Number($('[data-type-weight="' + checkbox.dataset.numberType + '"]').value);
  }
  return weights;
}

// 금액 자릿수에서 0보다 큰 상대 비율만 요청에 포함한다.
function selectedMoneyDigitWeights() {
  return Object.fromEntries([...document.querySelectorAll("[data-money-digit]")].filter((input) => Number(input.value) > 0).map((input) => [input.dataset.moneyDigit, Number(input.value)]));
}

// 이전 작업이나 초안의 자릿수 비율을 복원하며 기본값은 균등 비율이다.
function restoreMoneyDigitWeights(weights) {
  for (const input of document.querySelectorAll("[data-money-digit]")) input.value = weights ? weights[input.dataset.moneyDigit] || 0 : 1;
}

// 작업이나 입력 초안에 저장한 숫자 유형과 비율을 복원한다.
function restoreTypeWeights(weights) {
  for (const checkbox of document.querySelectorAll("[data-number-type]")) {
    const weight = $('[data-type-weight="' + checkbox.dataset.numberType + '"]');
    checkbox.checked = !weights || Object.hasOwn(weights, checkbox.dataset.numberType);
    weight.value = weights?.[checkbox.dataset.numberType] || 1;
    weight.disabled = !checkbox.checked;
  }
}

// 프롬프트 초안을 보존하고 입력 글자 수를 갱신한다.
function saveDraft() {
  remember("type-weights", JSON.stringify(selectedTypeWeights()));
  remember("money-digit-weights", JSON.stringify(selectedMoneyDigitWeights()));
  remember("prompt", $("#prompt").value);
  $("#character-count").textContent = $("#prompt").value.length.toLocaleString() + "자";
}

// 짧은 작업 결과를 알린다.
function toast(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 3500);
}

// 해당 단계의 오류만 표시한다.
function showError(selector, message) {
  $(selector).textContent = message;
  $(selector).hidden = !message;
}

// 검토를 마친 텍스트 선택을 후속 단계에서 그대로 사용한다.
function isSelected(row) {
  return row.accepted && !row.needs_review && !row.quality_issue;
}

// 현재 작업에서 사용하기로 한 문장 목록을 반환한다.
function selectedRows() {
  return (state.job?.examples || []).filter(isSelected);
}

// 목록에서 체크한 항목 중 준비된 유효 WAV만 저장 대상으로 반환한다.
function includedRows() {
  return generationRows().filter((row) => row.audio?.status === "ready");
}

// 텍스트 또는 음성이 생성 중인지 확인한다.
function isActive() {
  return Boolean(state.job && (ACTIVE.has(state.job.status) || ACTIVE.has(state.job.audio?.status) || state.job.auto_audio_status === "pending"));
}

// 스냅샷을 반영하고 실행 중에만 다음 조회를 예약한다.
function setJob(job) {
  clearTimeout(state.timer);
  if (state.job?.id !== job?.id) {
    state.filter = "all";
    for (const player of document.querySelectorAll("audio")) player.pause();
    clearWaveformCache();
    state.generationSelection = new Set();
    state.rowDrafts = {};
    state.audioRequestedIds = [];
    try { state.generationSelection = new Set(JSON.parse(recalled("generation-selection-" + job?.id)) || []); } catch (_) { /* 올바르지 않은 선택 기록은 무시한다. */ }
    $("#list-filter").value = "all";
    $("#process-batches").dataset.signature = "";
    state.saveResult = null;
  }
  state.job = job;
  if (job?.audio?.last_mapping) state.saveResult = job.audio.last_mapping;
  render();
  if (isActive()) state.timer = setTimeout(pollJob, 1100);
}

// 생성 진행 중 연결이 끊겨도 해당 작업의 조회만 재시도한다.
async function pollJob() {
  if (!state.job) return;
  const identifier = state.job.id;
  try {
    const job = await api("/api/jobs/" + identifier);
    if (state.job?.id !== identifier) return;
    state.connected = true;
    showError("#connection-notice", "");
    setJob(job);
    if (!isActive()) await loadHistory();
  } catch (error) {
    state.connected = false;
    showError("#connection-notice", error.message + " 다시 연결하는 중입니다.");
    renderControls();
    state.timer = setTimeout(pollJob, 3000);
  }
}

// 생성과 검토 버튼을 현재 상태에 맞춰 활성화한다.
function renderControls() {
  const active = isActive();
  const unavailable = !state.connected || active || state.busy;
  for (const id of ["prompt", "count", "batch-size", "output-root", "new-job", "job-select"]) $("#" + id).disabled = active || state.busy;
  for (const node of document.querySelectorAll("[data-number-type]")) node.disabled = active || state.busy;
  for (const node of document.querySelectorAll("[data-type-weight]")) {
    const selected = $('[data-number-type="' + node.dataset.typeWeight + '"]').checked;
    node.disabled = active || state.busy || !selected;
  }
  const moneySelected = $('[data-number-type="money"]').checked;
  $("#money-digit-settings").hidden = !moneySelected;
  for (const input of document.querySelectorAll("[data-money-digit]")) input.disabled = active || state.busy || !moneySelected;
  for (const node of document.querySelectorAll("[data-example]")) node.disabled = active || state.busy;
  $("#add-manual").disabled = unavailable;
  $("#upload-excel").disabled = unavailable;
  $("#excel-file").disabled = unavailable;
  $("#manual-save").disabled = unavailable;
  $("#manual-cancel").disabled = state.busy;
  for (const input of document.querySelectorAll("#manual-form textarea")) input.disabled = active || state.busy;
  $("#generate-button").disabled = unavailable;
  $("#delete-data").disabled = unavailable;
  $("#confirm-delete").disabled = unavailable || !state.deletionPreview?.files;
  $("#confirm-delete").textContent = state.deleting ? "삭제 중…" : "전체 삭제";
  $("#cancel-delete").disabled = state.deleting;
  $("#close-delete").disabled = state.deleting;
  $("#refresh-delete").disabled = state.busy || active;
  $("#generate-label").textContent = ACTIVE.has(state.job?.status) ? "텍스트 생성 중" : "발화 생성";
  $("#cancel-button").hidden = !ACTIVE.has(state.job?.status);
  $("#cancel-button").disabled = !state.connected || state.job?.status === "cancelling";
  $("#tts-generate").disabled = unavailable || !state.audioAvailable || !generationRows().length;
  $("#tts-generate").replaceChildren(document.createTextNode("선택한 " + generationRows().length + "개 음성 생성"), icon("arrow"));
  $("#select-valid").disabled = unavailable || !selectedRows().some((row) => !state.generationSelection.has(row.id));
  $("#clear-text").disabled = unavailable || !generationRows().length;
  $("#export-button").disabled = unavailable || !selectedRows().length;
  $("#tts-cancel").hidden = !ACTIVE.has(state.job?.audio?.status);
  $("#tts-cancel").disabled = !state.connected || state.job?.audio?.status === "cancelling";
  $("#retry-audio").hidden = active || !generationRows().some((row) => row.audio?.status !== "ready");
  $("#retry-audio").disabled = unavailable || !state.audioAvailable;
  $("#audio-output-root").disabled = active || state.busy;
  $("#create-mapping").disabled = unavailable || !includedRows().length;
  $("#create-mapping").replaceChildren(icon("down"), document.createTextNode(includedRows().length + "개 선택 데이터셋 저장"));
  for (const checkbox of document.querySelectorAll("[data-text-choice]")) {
    const row = state.job?.examples.find((item) => item.id === checkbox.dataset.textChoice);
    checkbox.disabled = unavailable || !row || !isSelected(row);
  }
  for (const control of document.querySelectorAll("[data-row-edit]")) control.disabled = unavailable;
  for (const button of document.querySelectorAll("[data-row-save]")) button.disabled = unavailable || !state.rowDrafts[button.dataset.rowSave];
  for (const button of document.querySelectorAll("[data-row-generate]")) {
    const row = state.job?.examples.find((item) => item.id === button.dataset.rowGenerate);
    button.disabled = unavailable || !state.audioAvailable || !row || !isSelected(row);
  }
}

// 단계 헤더와 진행·선택 수를 서버의 실제 상태로 표시한다.
function renderSummary() {
  const job = state.job;
  const rows = job?.examples || [];
  $("#workspace-title").textContent = job?.title || "새 데이터 작업";
  $("#workspace-status").textContent = isActive() ? "작업 중" : (STATUS_LABELS[job?.status] || "준비");
  $("#workspace-status").className = "badge " + (isActive() ? "review" : "ready");
  $("#text-count").textContent = rows.length;
  $("#selected-count").textContent = generationRows().length;
  $("#review-count").textContent = rows.filter((row) => row.needs_review).length;
  $("#text-selection-label").textContent = generationRows().length + "개 선택";
  $("#audio-selection-label").textContent = includedRows().length + "개 저장 예정";
  $("#generation-progress").hidden = !job;
  $("#progress-message").textContent = job?.message || "";
  $("#progress-count").textContent = job ? job.counts.accepted + " / " + job.settings.count : "";
  $("#progress-bar").max = job?.settings.count || 1;
  $("#progress-bar").value = job?.counts.accepted || 0;
  const runtime = job?.model_runtime;
  const runtimeText = runtime?.active_model === "llm" ? "LLM 처리 중 · TTS는 대기합니다." : runtime?.active_model === "tts" ? "LLM 실행 종료 · TTS 작업 진행 중" : runtime?.last_released === "tts" ? "TTS 프로세스 종료 · 모델 메모리 점유 해제" : runtime?.last_released === "llm" ? "LLM 프로세스 종료 · 모델 메모리 점유 해제" : "모델은 작업할 때만 로딩합니다.";
  $("#model-runtime-status").textContent = runtimeText;
  $("#audio-runtime-status").textContent = runtimeText;
  if (job?.auto_audio_error) showError("#text-alert", "자동 음성 생성 실패: " + job.auto_audio_error);
  const audio = job?.audio;
  const ready = selectedRows().filter((row) => row.audio?.status === "ready").length;
  const failed = selectedRows().filter((row) => row.audio?.status === "failed").length;
  $("#audio-status").textContent = (state.audioError || (ACTIVE.has(audio?.status) ? audio.message : "WAV " + ready + "개 준비")) + (failed ? " · 실패 " + failed + "개" : "");
  $("#audio-progress").hidden = !ACTIVE.has(audio?.status);
  const pending = selectedRows().filter((row) => state.audioRequestedIds.includes(row.id));
  $("#audio-progress").max = Math.max(1, pending.length || selectedRows().length);
  $("#audio-progress").value = pending.length ? pending.filter((row) => ["ready", "failed"].includes(row.audio?.status)).length : ready + failed;
  $("#save-result").hidden = !state.saveResult;
  $("#mapping-status").textContent = state.saveResult ? "마지막 저장 · " + state.saveResult.count + "개 데이터셋 완성" : "";
  $("#mapping-path").textContent = state.saveResult?.output_dir || "";
}

const NUMBER_TYPE_LABELS = {money: "금액", interest_rate: "금리", temperature: "온도", date: "날짜", time: "시간", phone: "전화번호", decimal: "소수", count: "수량", ordinal: "순서", ratio: "비율"};
const NUMBER_CONTEXT_LABELS = {amount: "금액", interest: "이율", celsius: "섭씨", calendar: "달력 날짜", clock: "시각", duration: "소요 시간", phone: "자릿수 읽기", measure: "소수", quantity: "수량", order: "순번", rank: "등수", percentage: "백분율"};
const PROCESS_STEPS = [["number", "숫자 생성", "코드"], ["template", "템플릿 생성", "LLM"], ["spoken", "한글 읽기", "LLM"], ["mapping", "매핑·검증", "코드"]];

// 서버와 동일한 최대 나머지 방식으로 유형별 목표를 표시한다.
function processQuotas(weights, count) {
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
  const entries = Object.entries(weights).map(([type, weight]) => ({type, quota: Math.floor(count * weight / total), remainder: count * weight % total}));
  const remaining = count - entries.reduce((sum, entry) => sum + entry.quota, 0);
  for (const entry of [...entries].sort((first, second) => second.remainder - first.remainder).slice(0, remaining)) entry.quota += 1;
  return entries;
}

// 템플릿의 숫자 자리만 강조하고 나머지 문장을 안전한 텍스트로 표시한다.
function templatePreview(template) {
  const preview = element("p", "template-preview");
  const parts = template.split("{number}");
  parts.forEach((part, index) => {
    if (index) preview.append(element("mark", "", "{number}"));
    preview.append(document.createTextNode(part));
  });
  return preview;
}

// 단계별 산출물 하나를 라벨과 함께 표시한다.
function processValue(label, value, className = "") {
  const block = element("div", "process-value " + className);
  block.append(element("span", "caption", label), element("p", value ? "" : "pending-value", value || "생성 대기 중"));
  return block;
}

// 생성 중인 항목에도 템플릿·숫자·읽기·매핑 결과를 순차적으로 채운다.
function processExample(item, index) {
  const card = element("article", "process-example");
  const heading = element("div", "process-example-heading");
  heading.append(element("strong", "", String(index + 1).padStart(2, "0") + " · " + (NUMBER_TYPE_LABELS[item.type] || item.type || "숫자")),
    element("span", "context-label", NUMBER_CONTEXT_LABELS[item.context] || item.context || ""));
  if (Object.hasOwn(item, "accepted")) heading.append(element("span", "badge " + (item.accepted ? "ready" : "review"), item.accepted ? "채택" : "미채택 · 검토 결과 확인"));
  card.append(heading);
  const template = element("div", "process-template");
  template.append(element("span", "caption", "02 · LLM 템플릿"), item.template ? templatePreview(item.template) : element("p", "pending-value", "템플릿 생성 중"));
  card.append(template);
  const values = element("div", "process-values");
  values.append(processValue("01 · 코드 생성 숫자", item.value, "number-value"), processValue("03 · LLM 한글 읽기", item.spoken, "spoken-value"));
  card.append(values);
  if (item.normalized_text || item.spoken_text) {
    const mapping = element("div", "process-mapping");
    mapping.append(element("span", "caption", "04 · 최종 매핑"), textPair(item));
    card.append(mapping);
  }
  if (item.quality_issue) card.append(element("p", "review-note", item.quality_issue));
  return card;
}

// 저장된 모든 배치의 중간 결과를 보존하고 최신 배치를 먼저 보여준다.
function renderProcessBatches(batches) {
  const container = $("#process-batches");
  const signature = JSON.stringify(batches);
  if (container.dataset.signature === signature) return;
  const opened = new Set([...container.querySelectorAll("details[open]")].map((node) => node.dataset.batch));
  const previous = new Set([...container.children].map((node) => node.dataset.batch));
  const nodes = [...batches].reverse().map((batch, index) => {
    const details = element("details", "process-batch");
    details.dataset.batch = String(batch.index);
    details.open = opened.has(String(batch.index)) || (!previous.has(String(batch.index)) && index === 0);
    const summary = element("summary");
    const labels = {running: "생성 중", completed: "처리 완료", failed: "오류 · 재시도 대상", cancelled: "중지됨", interrupted: "중단됨"};
    summary.append(document.createTextNode("배치 " + batch.index + " · " + batch.examples.length + "개"), element("span", "muted", labels[batch.status] || batch.status));
    details.append(summary);
    batch.examples.forEach((item, itemIndex) => details.append(processExample(item, itemIndex)));
    if (batch.error) details.append(element("p", "alert", batch.error));
    return details;
  });
  container.replaceChildren(...nodes);
  container.dataset.signature = signature;
}

// 이전 작업의 템플릿과 발화가 일치할 때 숫자에 해당하는 한글 읽기만 복원한다.
function previousSpokenNumber(row) {
  const [prefix, suffix] = row.template.split("{number}");
  if (suffix === undefined || !row.spoken_text?.startsWith(prefix) || !row.spoken_text.endsWith(suffix)) return "이전 작업에 읽기 원본이 없습니다.";
  return row.spoken_text.slice(prefix.length, suffix ? -suffix.length : undefined);
}

// 작업의 실제 단계와 유형별 채택량을 표시하며 기존 작업은 저장된 정보로 복원한다.
function renderGenerationProcess() {
  const job = state.job;
  const panel = $("#generation-process");
  panel.hidden = !job?.settings.type_weights && !job?.generation_batches?.length;
  if (panel.hidden) return;
  if (panel.dataset.job !== job.id) {
    panel.dataset.job = job.id;
    panel.open = !job.examples.length && ACTIVE.has(job.status);
  }
  let batches = job.generation_batches || [];
  if (!batches.length && job.examples.some((row) => row.template)) {
    batches = [{index: 1, status: "completed", phase: "completed", examples: job.examples.filter((row) => row.template).map((row) => ({...row,
      context: row.number_context, value: row.generated_number?.value, spoken: previousSpokenNumber(row)}))}];
  }
  const current = batches.at(-1);
  const active = ACTIVE.has(job.status);
  const phaseIndex = PROCESS_STEPS.findIndex(([phase]) => phase === (current?.failed_phase || current?.phase));
  const badge = $("#process-status");
  badge.textContent = active ? "진행 중 · 배치 " + (current?.index || 1) : (STATUS_LABELS[job.status] || job.status);
  badge.className = "badge " + (job.status === "completed" ? "ready" : "review");
  $("#process-steps").replaceChildren(...PROCESS_STEPS.map(([phase, label, owner], index) => {
    const done = current?.phase === "completed" || (phaseIndex >= 0 && index < phaseIndex);
    const running = active && current?.status === "running" && index === phaseIndex;
    const stopped = !running && current && current.phase !== "completed" && (!active || current.status !== "running") && index === phaseIndex;
    const step = element("li", "process-step" + (done ? " is-done" : "") + (running ? " is-running" : "") + (stopped ? " is-stopped" : ""));
    if (running) step.setAttribute("aria-current", "step");
    step.append(element("span", "process-step-number", done ? "✓" : String(index + 1)), element("strong", "", label), element("small", "", owner + " · " + (done ? "완료" : running ? "진행 중" : stopped ? "중단·오류" : "대기")));
    return step;
  }));
  const quotas = processQuotas(job.settings.type_weights || {}, job.settings.count);
  $("#process-quotas").replaceChildren(...quotas.map(({type, quota}) => {
    const accepted = job.examples.filter((row) => row.accepted && row.type === type).length;
    return element("span", "quota-chip", (NUMBER_TYPE_LABELS[type] || type) + " " + accepted + " / " + quota);
  }));
  $("#process-batch-count").textContent = batches.length + "개 배치 · 최신순";
  renderProcessBatches(batches);
}

// 현재 목록 필터에 해당하는 텍스트만 반환한다.
function visibleRows() {
  return (state.job?.examples || []).filter((row) => {
    const matches = state.filter === "all" || (state.filter === "accepted" && isSelected(row)) || (state.filter === "selected" && state.generationSelection.has(row.id)) || (state.filter === "review" && row.needs_review) || (state.filter === "rejected" && !row.accepted && !row.needs_review);
    return matches;
  });
}

// 발화와 정답을 읽기 쉬운 두 열로 표시한다.
function textPair(row) {
  const pair = element("div", "text-pair");
  for (const [label, text, className] of [["읽을 발화", row.spoken_text, ""], ["정답 표기", row.normalized_text, " answer"]]) {
    const column = element("div", "text-column" + className);
    column.append(element("span", "caption", label), element("p", "", text || "생성 원문과 실패 사유를 확인하세요."));
    pair.append(column);
  }
  return pair;
}

// 표현 확인이 필요한 이유를 사용자가 읽을 수 있게 전달한다.
function reviewReason(row) {
  if (!row.spoken_text) return row.quality_issue || row.candidate?.number_generation_issue || row.reason || "생성 형식이 맞지 않습니다. 원문을 확인하고 다시 생성하세요.";
  if (row.quality_issue) return row.quality_issue;
  if (row.needs_review && row.review_note) return row.review_note;
  if (row.reason === "near_duplicate") return "비슷한 문장이 있습니다. 필요한 변형인지 확인하고 선택하세요.";
  if (row.reason === "structure_duplicate") return "숫자·고유명사만 바뀌었거나 문장 형식이 반복되어 자동 제외했습니다.";
  if (row.reason === "transcript_conflict") return "같은 발화의 정답이 서로 다릅니다. 수정한 뒤 사용하세요.";
  if (row.needs_review) return "발화와 정답을 확인한 뒤 사용 여부를 선택하세요.";
  return "";
}

// 사용자가 음성을 만들려고 체크한 유효 발화만 반환한다.
function generationRows() {
  return selectedRows().filter((row) => state.generationSelection.has(row.id));
}

// 작업별 음성 생성 선택을 브라우저에 보존한다.
function rememberGenerationSelection() {
  remember("generation-selection-" + state.job.id, JSON.stringify([...state.generationSelection]));
}

// 다른 작업으로 이동할 때 진행 중인 파형 요청과 캐시를 정리한다.
function clearWaveformCache() {
  waveformObserver?.disconnect();
  waveformObserver = null;
  for (const controller of waveformControllers.values()) controller.abort();
  waveformControllers.clear();
  waveformCache.clear();
}

// RIFF 청크를 확인하고 PCM16 샘플의 실제 최대 진폭을 구간별로 계산한다.
function wavPeaks(buffer, bins = 96) {
  const view = new DataView(buffer);
  const word = (offset) => String.fromCharCode(...new Uint8Array(buffer, offset, 4));
  if (buffer.byteLength < 12 || word(0) !== "RIFF" || word(8) !== "WAVE") throw new Error("WAV 형식이 아닙니다.");
  const end = view.getUint32(4, true) + 8;
  if (end > buffer.byteLength || end < 12) throw new Error("WAV 파일이 불완전합니다.");
  let format = null;
  let data = null;
  for (let offset = 12; offset + 8 <= end;) {
    const length = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (start + length > end) throw new Error("WAV 청크가 불완전합니다.");
    if (word(offset) === "fmt " && length >= 16) format = {codec: view.getUint16(start, true), channels: view.getUint16(start + 2, true), rate: view.getUint32(start + 4, true), align: view.getUint16(start + 12, true), bits: view.getUint16(start + 14, true)};
    if (word(offset) === "data") data = {start, length};
    offset = start + length + (length % 2);
  }
  if (!format || format.codec !== 1 || format.bits !== 16 || !format.channels || !format.rate || format.align !== format.channels * 2 || !data || !data.length || data.length % format.align) throw new Error("PCM16 WAV만 파형을 표시할 수 있습니다.");
  const frames = data.length / format.align;
  const peaks = Array(Math.min(bins, frames)).fill(0);
  for (let frame = 0; frame < frames; frame++) {
    const bin = Math.min(peaks.length - 1, Math.floor(frame * peaks.length / frames));
    for (let channel = 0; channel < format.channels; channel++) peaks[bin] = Math.max(peaks[bin], Math.abs(view.getInt16(data.start + frame * format.align + channel * 2, true)) / 32768);
  }
  return {peaks, duration: frames / format.rate};
}

// 같은 버전의 음성 파형 계산을 공유하고 캐시 크기를 제한한다.
function loadWaveform(url) {
  if (!waveformCache.has(url)) {
    const controller = new AbortController();
    waveformControllers.set(url, controller);
    const pending = fetch(url, {signal: controller.signal}).then((response) => {
      if (!response.ok) throw new Error("WAV를 불러오지 못했습니다.");
      return response.arrayBuffer();
    }).then(wavPeaks).catch((error) => { waveformCache.delete(url); throw error; }).finally(() => {
      if (waveformControllers.get(url) === controller) waveformControllers.delete(url);
    });
    waveformCache.set(url, pending);
    if (waveformCache.size > 128) waveformCache.delete(waveformCache.keys().next().value);
  }
  return waveformCache.get(url);
}

// 음성 길이를 분과 초로 표시한다.
function audioTime(value) {
  const seconds = Math.max(0, Math.floor(Number.isFinite(value) ? value : 0));
  return Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
}

// 준비된 WAV에 재생·탐색 가능한 실제 파형을 붙인다.
function inlinePlayer(row) {
  const container = element("div", "inline-player");
  const audio = element("audio");
  audio.preload = "none";
  const source = row.audio.url;
  audio.src = source.startsWith("data:") ? source : source + (source.includes("?") ? "&" : "?") + "v=" + encodeURIComponent(row.audio.fingerprint || "");
  const button = element("button", "play-button");
  button.type = "button";
  const wave = element("div", "waveform");
  wave.tabIndex = 0;
  wave.setAttribute("role", "slider");
  wave.setAttribute("aria-label", "재생 위치");
  wave.setAttribute("aria-valuemin", "0");
  wave.setAttribute("aria-valuemax", "100");
  wave.setAttribute("aria-valuenow", "0");
  wave.append(element("span", "waveform-message", "파형 불러오는 중…"));
  const time = element("span", "audio-time");
  let duration = row.audio.duration_seconds || 0;
  let loaded = false;
  const update = () => {
    const progress = duration ? Math.min(1, audio.currentTime / duration) : 0;
    wave.style.setProperty("--played", (progress * 100) + "%");
    wave.setAttribute("aria-valuenow", String(Math.round(progress * 100)));
    wave.setAttribute("aria-valuetext", audioTime(audio.currentTime) + " / " + audioTime(duration));
    time.textContent = audioTime(audio.currentTime) + " / " + audioTime(duration);
    button.setAttribute("aria-label", audio.paused ? "재생" : "일시 정지");
    button.replaceChildren(icon(audio.paused ? "play" : "pause"));
  };
  const seek = (seconds) => {
    if (!Number.isFinite(audio.duration) || !audio.duration) return;
    audio.currentTime = Math.max(0, Math.min(audio.duration, seconds));
    update();
  };
  wave.addEventListener("click", (event) => {
    const box = wave.getBoundingClientRect();
    seek((event.clientX - box.left) / box.width * duration);
  });
  wave.addEventListener("keydown", (event) => {
    const targets = {ArrowRight: audio.currentTime + 1, ArrowLeft: audio.currentTime - 1, Home: 0, End: duration};
    if (!Object.hasOwn(targets, event.key)) return;
    event.preventDefault();
    seek(targets[event.key]);
  });
  button.addEventListener("click", async () => {
    if (!audio.paused) { audio.pause(); return; }
    for (const other of document.querySelectorAll("audio")) if (other !== audio) other.pause();
    try { await audio.play(); } catch (_) { toast("음성을 재생하지 못했습니다. WAV 파일을 확인하세요."); }
  });
  for (const name of ["timeupdate", "play", "pause", "ended"]) audio.addEventListener(name, update);
  audio.addEventListener("loadedmetadata", () => { duration = audio.duration; update(); });
  audio.addEventListener("error", () => { button.disabled = true; button.title = "WAV 파일을 불러오지 못했습니다."; });
  const draw = async () => {
    if (loaded) return;
    loaded = true;
    try {
      const result = await loadWaveform(audio.src);
      if (!wave.isConnected) return;
      duration = result.duration;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 1000 48");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("aria-hidden", "true");
      const path = document.createElementNS(svg.namespaceURI, "path");
      path.setAttribute("d", result.peaks.map((peak, index) => "M" + ((index + 0.5) * 1000 / result.peaks.length) + " " + (24 - peak * 23) + "v" + Math.max(0.5, peak * 46)).join(" "));
      svg.append(path);
      const progress = svg.cloneNode(true);
      progress.classList.add("waveform-played");
      wave.replaceChildren(svg, progress);
      wave.dataset.loaded = "true";
      update();
    } catch (error) {
      if (error.name !== "AbortError" && wave.isConnected) wave.replaceChildren(element("span", "waveform-message", "파형을 불러오지 못했습니다."));
    }
  };
  wave.loadWaveform = draw;
  if (!waveformObserver) waveformObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) if (entry.isIntersecting) { waveformObserver.unobserve(entry.target); entry.target.loadWaveform(); }
  }, {root: $(".utterance-scroll"), rootMargin: "100px"});
  waveformObserver.observe(wave);
  container.append(button, wave, time, audio);
  update();
  return container;
}

// 항상 표시되는 입력창에서 발화와 정답을 수정하고 저장·취소한다.
function editableTextPair(row) {
  const pair = element("div", "text-pair inline-text-editor");
  const draft = state.rowDrafts[row.id];
  const fields = {};
  const toolbar = element("div", "inline-edit-actions");
  const accepted = element("input");
  accepted.type = "checkbox";
  accepted.checked = draft?.accepted ?? isSelected(row);
  accepted.dataset.rowEdit = row.id;
  const review = element("label", "checkbox-label");
  review.append(accepted, document.createTextNode("내용 확인 · 사용 허용"));
  const save = element("button", "button secondary", "저장");
  save.type = "button";
  save.dataset.rowSave = row.id;
  const cancel = element("button", "text-button", "취소");
  cancel.type = "button";
  cancel.dataset.rowEdit = row.id;
  const capture = () => {
    state.rowDrafts[row.id] = {spoken: fields.spoken.value, normalized: fields.normalized.value, accepted: accepted.checked};
    toolbar.hidden = false;
    renderControls();
  };
  const reset = () => {
    delete state.rowDrafts[row.id];
    const current = state.job.examples.find((item) => item.id === row.id);
    for (const [key, property] of [["spoken", "spoken_text"], ["normalized", "normalized_text"]]) {
      fields[key].value = current[property] || "";
    }
    accepted.checked = isSelected(current);
    review.hidden = isSelected(current);
    toolbar.hidden = isSelected(current);
    renderControls();
  };
  for (const [key, title, property] of [["spoken", "읽을 발화", "spoken_text"], ["normalized", "정답 표기", "normalized_text"]]) {
    const column = element("div", "text-column" + (key === "normalized" ? " answer" : ""));
    const input = element("input", "inline-text-input");
    input.type = "text";
    input.value = draft?.[key] ?? row[property] ?? "";
    input.dataset.rowEdit = row.id;
    input.dataset.editField = key;
    input.setAttribute("aria-label", title);
    fields[key] = input;
    input.addEventListener("input", capture);
    input.addEventListener("keydown", (event) => {
      if (event.isComposing) return;
      if (event.key === "Escape") { event.preventDefault(); reset(); input.focus(); }
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); save.click(); }
    });
    column.append(element("span", "caption", title), input);
    pair.append(column);
  }
  accepted.addEventListener("input", capture);
  cancel.addEventListener("click", () => { reset(); fields.spoken.focus(); });
  save.addEventListener("click", async () => {
    const draft = state.rowDrafts[row.id];
    if (!draft) return;
    if (await updateText(state.job.examples.find((item) => item.id === row.id), draft.accepted, draft.spoken, draft.normalized)) {
      delete state.rowDrafts[row.id];
      const card = [...$("#text-list").children].find((node) => node.dataset.rowId === row.id);
      card?.querySelector(".inline-text-editor")?.cancelEditing();
      render();
    }
  });
  review.hidden = isSelected(row);
  toolbar.hidden = !draft && isSelected(row);
  toolbar.append(review, save, cancel, element("span", "edit-shortcut muted", "⌘/Ctrl+Enter 저장 · Esc 취소"));
  pair.append(toolbar);
  pair.cancelEditing = reset;
  return pair;
}

// 발화·정답·음성 작업을 목록의 한 행에 모은다.
function utteranceRow(row, index) {
  const card = element("article", "utterance-row");
  card.dataset.rowId = row.id;
  const choice = element("input", "utterance-checkbox");
  choice.type = "checkbox";
  choice.dataset.textChoice = row.id;
  choice.setAttribute("aria-label", (index + 1) + "번 발화 생성·저장 선택");
  choice.addEventListener("change", () => {
    if (choice.checked) state.generationSelection.add(row.id);
    else state.generationSelection.delete(row.id);
    rememberGenerationSelection();
    render();
  });
  const content = element("div", "utterance-content");
  const heading = element("div", "row-meta");
  heading.append(element("span", "muted", String(index + 1).padStart(2, "0")), element("span", "badge", row.source === "manual" ? "직접 입력" : (NUMBER_TYPE_LABELS[row.type] || row.type || "일반")));
  const status = element("span", "badge row-validation");
  heading.append(status);
  const pair = editableTextPair(row);
  const actions = element("div", "row-actions");
  const generate = element("button", "text-button", "음성 생성");
  generate.type = "button";
  generate.dataset.rowGenerate = row.id;
  generate.addEventListener("click", () => startAudio([row.id]));
  if (row.audio?.status !== "ready") actions.append(generate);
  content.append(heading, pair);
  if (row.audio?.status === "ready" && row.audio.url) content.append(inlinePlayer(row));
  else if (row.audio?.status && row.audio.status !== "missing") content.append(element("p", "row-audio-status", row.audio.error || row.audio.message || ({queued: "음성 생성 대기", running: "음성 생성 중", generating: "음성 생성 중", failed: "음성 생성 실패"}[row.audio.status] || row.audio.status)));
  const reason = reviewReason(row);
  if (reason) content.append(element("p", "review-note", reason));
  content.append(actions);
  if (!row.spoken_text && row.candidate) content.append(failedSource(row.candidate));
  card.append(choice, content);
  return card;
}

// 실패한 원문을 보존해 사용자가 실제 거절 사유와 함께 확인할 수 있게 한다.
function failedSource(candidate) {
  const details = element("details", "failed-source");
  details.append(element("summary", "", "생성 원문 보기"), element("pre", "", JSON.stringify(candidate, null, 2)));
  return details;
}

// 변경된 행만 교체해 선택·자동 조회 중에도 재생과 입력을 유지한다.
function syncRows(rows) {
  const list = $("#text-list");
  const current = new Map([...list.children].map((node) => [node.dataset.rowId, node]));
  const wanted = new Set(rows.map((row) => row.id));
  const discard = (node) => {
    for (const audio of node.querySelectorAll("audio")) audio.pause();
    for (const wave of node.querySelectorAll(".waveform")) waveformObserver?.unobserve(wave);
    node.remove();
  };
  for (const node of current.values()) if (!wanted.has(node.dataset.rowId)) discard(node);
  rows.forEach((row, index) => {
    const signature = JSON.stringify([state.job?.id, row.spoken_text, row.normalized_text, row.type, row.quality_issue, row.candidate, row.reason, row.review_note, row.audio?.status, row.audio?.url, row.audio?.fingerprint, row.audio?.error]);
    let node = current.get(row.id);
    if (!node || node.dataset.signature !== signature) {
      if (node) discard(node);
      node = utteranceRow(row, index);
      node.dataset.signature = signature;
    }
    if (list.children[index] !== node) list.insertBefore(node, list.children[index] || null);
    node.querySelector("[data-text-choice]").checked = state.generationSelection.has(row.id);
    node.classList.toggle("is-selected", state.generationSelection.has(row.id));
    const badge = node.querySelector(".row-validation");
    badge.textContent = isSelected(row) ? "생성 가능" : row.needs_review ? "확인 필요" : "제외";
  });
}

// 설정과 발화 목록을 같은 작업 화면에서 갱신한다.
function render() {
  const settings = $("#setup-stage");
  const jobKey = state.job?.id || "new";
  if (settings.dataset.job !== jobKey) {
    settings.dataset.job = jobKey;
    settings.classList.toggle("is-collapsed", Boolean(state.job));
    $("#toggle-settings").setAttribute("aria-expanded", String(!state.job));
    $("#toggle-settings").textContent = state.job ? "펼치기" : "접기";
  }
  renderSummary();
  renderGenerationProcess();
  const rows = visibleRows();
  $("#text-empty").hidden = Boolean(state.job?.examples.length);
  $("#text-no-results").hidden = !state.job?.examples.length || rows.length > 0;
  $("#text-empty h3").textContent = isActive() ? "발화를 생성하고 있습니다" : "첫 발화를 만들어보세요";
  $("#list-filter").value = state.filter;
  syncRows(rows);
  renderControls();
}

// 새 작업이나 현재 목록에 직접 입력할 대상을 고정하고 폼을 연다.
function openManualForm() {
  if (!state.connected || state.busy || isActive()) return;
  state.manualJobId = state.job?.id || null;
  $("#manual-form").hidden = false;
  $("#add-manual").setAttribute("aria-expanded", "true");
  $("#manual-target").textContent = state.job ? "현재 작업에 추가" : "새 작업으로 저장";
  showError("#manual-alert", "");
  $("#manual-spoken").focus();
}

// 취소하거나 저장을 마친 직접 입력 폼을 정리한다.
function closeManualForm() {
  $("#manual-form").reset();
  $("#manual-form").hidden = true;
  $("#add-manual").setAttribute("aria-expanded", "false");
  showError("#manual-alert", "");
}

// 같은 줄의 발화와 정답을 검증해 목록에 추가하고 새 항목을 선택한다.
async function submitManualForm(event) {
  event.preventDefault();
  if (!state.connected || state.busy || isActive() || !$("#manual-form").reportValidity()) return;
  showError("#manual-alert", "");
  const spoken = $("#manual-spoken").value.trim().split(/\r?\n/).map((line) => line.trim());
  const transcripts = $("#manual-transcript").value.trim().split(/\r?\n/).map((line) => line.trim());
  if (spoken.length !== transcripts.length || spoken.some((line) => !line) || transcripts.some((line) => !line)) {
    showError("#manual-alert", "빈 줄 없이 발화와 정답을 같은 줄 수로 입력하세요.");
    return;
  }
  if ((state.job?.id || null) !== state.manualJobId) {
    showError("#manual-alert", "추가할 작업이 바뀌었습니다. 직접 추가를 다시 눌러 작업을 확인하세요.");
    return;
  }
  state.busy = true;
  renderControls();
  const previous = new Set((state.job?.examples || []).map((row) => row.id));
  try {
    const path = state.manualJobId ? "/api/jobs/" + state.manualJobId + "/examples" : "/api/jobs/manual";
    const payload = {examples: spoken.map((text, index) => ({spoken_text: text, normalized_text: transcripts[index]})), output_root: $("#output-root").value};
    const job = await api(path, {method: "POST", body: JSON.stringify(payload)});
    setJob(job);
    for (const row of job.examples) if (!previous.has(row.id) && isSelected(row)) state.generationSelection.add(row.id);
    state.filter = "all";
    rememberGenerationSelection();
    closeManualForm();
    render();
    try { await loadHistory(); } catch (_) { toast("발화는 저장됐습니다. 작업 목록은 다시 불러와 주세요."); }
  } catch (error) { showError("#manual-alert", error.message); }
  finally { state.busy = false; render(); }
}

// 업로드 파일을 JSON 요청에 담을 수 있는 Base64로 읽는다.
function readUploadFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(new Error("파일을 읽지 못했습니다. 파일을 다시 선택하세요."));
    reader.onabort = () => reject(new Error("파일 읽기가 취소되었습니다."));
    reader.readAsDataURL(file);
  });
}

// 첫 시트의 발화·정답을 현재 목록 또는 새 작업에 추가한다.
async function uploadExcel(event) {
  const file = event.target.files?.[0];
  if (!file || !state.connected || state.busy || isActive()) return;
  showError("#text-alert", "");
  if (!/\.xlsx$/i.test(file.name) || file.size > 5 * 1024 * 1024 || !file.size) {
    showError("#text-alert", "비어 있지 않은 5MB 이하 .xlsx 파일을 선택하세요. A열은 발화문, B열은 정답지입니다.");
    event.target.value = "";
    return;
  }
  const jobId = state.job?.id || null;
  const previous = new Set((state.job?.examples || []).map((row) => row.id));
  state.busy = true;
  renderControls();
  try {
    const encoded = await readUploadFile(file);
    if ((state.job?.id || null) !== jobId) throw new Error("추가할 작업이 바뀌었습니다. 엑셀 파일을 다시 선택하세요.");
    const path = jobId ? "/api/jobs/" + jobId + "/examples" : "/api/jobs/manual";
    const job = await api(path, {method: "POST", body: JSON.stringify({filename: file.name, excel_base64: encoded, output_root: $("#output-root").value})});
    setJob(job);
    for (const row of job.examples) if (!previous.has(row.id) && isSelected(row)) state.generationSelection.add(row.id);
    state.filter = "all";
    rememberGenerationSelection();
    render();
    toast("엑셀 발화 " + (job.examples.length - previous.size) + "개를 추가했습니다.");
    try { await loadHistory(); } catch (_) { toast("발화는 저장됐습니다. 작업 목록은 다시 불러와 주세요."); }
  } catch (error) { showError("#text-alert", error.message); }
  finally { event.target.value = ""; state.busy = false; render(); }
}

// 생성 요청에 이번 주제와 공통 규칙, 저장 경로를 함께 고정한다.
async function startGeneration(event) {
  event.preventDefault();
  if (state.busy || isActive() || !$("#generation-form").reportValidity()) return;
  state.busy = true;
  showError("#form-alert", "");
  renderControls();
  try {
    const typeWeights = selectedTypeWeights();
    if (!Object.keys(typeWeights).length) throw new Error("숫자 유형을 하나 이상 선택하세요.");
    const moneyDigitWeights = Object.hasOwn(typeWeights, "money") ? selectedMoneyDigitWeights() : undefined;
    if (moneyDigitWeights && !Object.keys(moneyDigitWeights).length) throw new Error("금액 자릿수 비율을 하나 이상 1 이상으로 입력하세요.");
    const payload = { money_digit_weights: moneyDigitWeights, auto_audio: false, type_weights: typeWeights, prompt: $("#prompt").value, output_root: $("#output-root").value, count: Number($("#count").value), batch_size: Number($("#batch-size").value), prompt_mode: true };
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
    state.saveResult = null;
    setJob(job);
    await loadHistory();
  } catch (error) { showError("#form-alert", error.message); }
  finally { state.busy = false; render(); }
}

// 현재 텍스트 생성만 중지한다.
async function cancelGeneration() {
  try { setJob(await api("/api/jobs/" + state.job.id + "/cancel", { method: "POST", body: "{}" })); }
  catch (error) { showError("#text-alert", error.message); }
}

// 한 문장의 사용 여부와 수정을 저장한다.
async function updateText(row, accepted, spoken = row.spoken_text, normalized = row.normalized_text, errorTarget = null) {
  if (state.busy || isActive()) return false;
  state.busy = true;
  const target = errorTarget || "#text-alert";
  showError(target, "");
  renderControls();
  try {
    setJob(await api("/api/jobs/" + state.job.id + "/examples/" + row.id, { method: "PATCH", body: JSON.stringify({ accepted, spoken_text: spoken, normalized_text: normalized }) }));
    return true;
  } catch (error) { showError(target, error.message); return false; }
  finally { state.busy = false; render(); }
}

// 생성 가능한 발화의 체크 선택만 변경하며 서버의 채택 정보는 보존한다.
async function bulkText(selected) {
  if (!state.job || state.busy || isActive()) return;
  state.generationSelection = new Set(selected ? selectedRows().map((row) => row.id) : []);
  rememberGenerationSelection();
  render();
}

// 텍스트 선택을 그대로 음성 생성에 넘기고 다음 단계로 이동한다.
async function startAudio(identifiers = null) {
  if (state.busy || isActive() || !selectedRows().length) return;
  state.busy = true;
  const errorTarget = "#audio-alert";
  showError(errorTarget, "");
  renderControls();
  try {
    const rows = Array.isArray(identifiers) ? selectedRows().filter((row) => identifiers.includes(row.id)) : generationRows();
    if (!rows.length) return;
    state.audioRequestedIds = rows.map((row) => row.id);
    const payload = { example_ids: rows.map((row) => row.id), output_root: $("#output-root").value };
    const job = await api("/api/jobs/" + state.job.id + "/audio", { method: "POST", body: JSON.stringify(payload) });
    setJob(job);
  } catch (error) { showError(errorTarget, error.message); }
  finally { state.busy = false; render(); }
}

// 진행 중인 음성 생성만 중지한다.
async function cancelAudio() {
  try { setJob(await api("/api/jobs/" + state.job.id + "/audio/cancel", { method: "POST", body: "{}" })); }
  catch (error) { showError("#audio-alert", error.message); }
}

// 최종 음성 선택과 정답을 자동으로 연결하여 새 데이터셋으로 저장한다.
async function createMapping() {
  if (state.busy || isActive() || !includedRows().length) return;
  state.busy = true;
  showError("#audio-alert", "");
  renderControls();
  try {
    state.saveResult = await api("/api/jobs/" + state.job.id + "/mapping", { method: "POST", body: JSON.stringify({ example_ids: includedRows().map((row) => row.id), include_all_ready: true, output_root: $("#audio-output-root").value }) });
    toast("선택한 음성과 정답을 저장했습니다.");
  } catch (error) { showError("#audio-alert", error.message); }
  finally { state.busy = false; render(); }
}

// 저장 루트 입력 두 곳을 동일한 값으로 유지한다.
function syncRoot(value) {
  $("#output-root").value = value;
  $("#audio-output-root").value = value;
  remember("output-root", value);
}

// 작업 선택 시 실행 당시 요구사항을 복원한다.
function fillForm(settings) {
  restoreTypeWeights(settings.type_weights);
  restoreMoneyDigitWeights(settings.money_digit_weights);
  $("#prompt").value = settings.prompt;
  $("#batch-size").value = [1, 3, 5, 10].includes(settings.batch_size) ? settings.batch_size : 3;
  $("#count").value = settings.count;
  syncRoot(settings.output_root || state.preferences?.output_root || "/tmp/ai-dataset/");
  saveDraft();
}

// 저장된 작업 목록을 본문 없이 갱신한다.
async function loadHistory() {
  const { jobs } = await api("/api/jobs");
  $("#job-select").replaceChildren(new Option("저장된 작업", ""));
  for (const job of jobs) $("#job-select").add(new Option((STATUS_LABELS[job.status] || job.status) + " · " + job.title, job.id));
  $("#job-select").value = state.job?.id || "";
  return jobs;
}

// 이전 데이터를 유지한 채 새 요구사항 입력으로 돌아간다.
function newJob() {
  if (state.busy || isActive()) return;
  for (const player of document.querySelectorAll("audio")) player.pause();
  setJob(null);
  $("#job-select").value = "";
  $("#batch-size").value = "3";
  for (const id of ["form-alert", "text-alert", "audio-alert"]) showError("#" + id, "");
  $("#prompt").focus();
}

// 저장된 작업의 텍스트·음성 단계와 제외 결정을 복원한다.
async function selectHistory() {
  const identifier = $("#job-select").value;
  if (!identifier) { newJob(); return; }
  try {
    const job = await api("/api/jobs/" + identifier);
    fillForm(job.settings);
    if (job.audio?.last_mapping?.output_root) syncRoot(job.audio.last_mapping.output_root);
    setJob(job);
  } catch (error) { showError("#connection-notice", error.message); }
}

// 텍스트만 필요한 경우 기존 JSONL 내보내기를 제공한다.
function exportDataset() {
  if (!state.job || !selectedRows().length) return;
  const link = element("a");
  link.href = "/api/jobs/" + state.job.id + "/selected-export";
  link.download = "dataset-" + state.job.id.slice(0, 8) + ".jsonl";
  document.body.append(link);
  link.click();
  link.remove();
}

// 실제 파일을 지우지 않고 삭제 대상 개수와 경로를 확인 창에 표시한다.
async function previewDeletion() {
  if (state.busy || isActive()) return;
  state.busy = true;
  state.deletionPreview = null;
  showError("#delete-alert", "");
  $("#refresh-delete").hidden = true;
  $("#delete-skipped").hidden = true;
  $("#delete-summary").replaceChildren(element("li", "", "삭제 대상을 확인하고 있습니다."));
  $("#delete-roots").replaceChildren();
  if (!$("#delete-dialog").open) $("#delete-dialog").showModal();
  $("#cancel-delete").focus();
  renderControls();
  try {
    const preview = await api("/api/data/deletion-preview", { method: "POST", body: "{}" });
    state.deletionPreview = preview;
    const labels = [["저장된 작업", preview.jobs], ["생성 항목", preview.examples], ["WAV 파일", preview.wav_files], ["매핑 데이터셋", preview.datasets], ["전체 파일", preview.files]];
    $("#delete-summary").replaceChildren(...labels.map(([label, count]) => element("li", "", label + " " + count.toLocaleString() + "개")));
    $("#delete-roots").replaceChildren(...preview.roots.map((path) => element("li", "", path)));
    if (!preview.files) $("#delete-summary").append(element("li", "", "삭제할 생성 파일이 없습니다."));
    $("#delete-skipped").hidden = !preview.skipped;
    $("#delete-skipped").textContent = preview.skipped + "개 경로는 생성 여부를 확인하지 못했거나 보호 대상으로 확인되어 제외했습니다.";
  } catch (error) { showError("#delete-alert", error.message); $("#refresh-delete").hidden = false; }
  finally { state.busy = false; renderControls(); }
}

// 삭제한 작업의 브라우저 단계 기록만 정리하고 사용자 입력·공통 규칙은 보존한다.
function forgetJobStages() {
  try {
    for (const key of Object.keys(localStorage)) if (key.startsWith("asr-studio-stage-") || key.startsWith("asr-studio-generation-selection-")) localStorage.removeItem(key);
  } catch (_) { /* 파일과 서버 작업의 삭제 결과는 유지한다. */ }
}

// 사용자가 확인한 최신 대상만 삭제하고 작업 화면을 초기 단계로 돌린다.
async function deleteAllData() {
  if (state.busy || isActive() || !state.deletionPreview?.files) return;
  state.busy = true;
  state.deleting = true;
  showError("#delete-alert", "");
  renderControls();
  for (const player of document.querySelectorAll("audio")) player.pause();
  try {
    const result = await api("/api/data/delete-all", { method: "POST", body: JSON.stringify({ token: state.deletionPreview.token, confirmed: true }) });
    forgetJobStages();
    state.saveResult = null;
    state.editingId = null;
    setJob(null);
    await loadHistory();
    state.deletionPreview = null;
    $("#delete-dialog").close();
    toast("생성 데이터 " + result.deleted_files + "개 파일을 삭제했습니다.");
  } catch (error) {
    state.deletionPreview = null;
    showError("#delete-alert", error.message);
    $("#refresh-delete").hidden = false;
    try {
      const jobs = await loadHistory();
      if (state.job && !jobs.some((job) => job.id === state.job.id)) setJob(null);
    } catch (_) { /* 삭제 오류와 재확인 버튼을 유지한다. */ }
  } finally { state.busy = false; state.deleting = false; render(); }
}

// 사용자 동작을 단계별 기능에 연결한다.
function bindEvents() {
  $("#generation-form").addEventListener("submit", startGeneration);
  $("#add-manual").addEventListener("click", openManualForm);
  $("#upload-excel").addEventListener("click", () => $("#excel-file").click());
  $("#excel-file").addEventListener("change", uploadExcel);
  $("#number-types").addEventListener("change", () => { renderControls(); saveDraft(); });
  $("#money-digit-settings").addEventListener("input", saveDraft);
  $("#prompt").addEventListener("input", saveDraft);
  for (const button of document.querySelectorAll("[data-example]")) button.addEventListener("click", () => { $("#prompt").value = EXAMPLES[button.dataset.example]; saveDraft(); $("#prompt").focus(); });
  $("#list-filter").addEventListener("change", () => { state.filter = $("#list-filter").value; render(); });
  $("#toggle-settings").addEventListener("click", () => {
    const collapsed = $("#setup-stage").classList.toggle("is-collapsed");
    $("#toggle-settings").setAttribute("aria-expanded", String(!collapsed));
    $("#toggle-settings").textContent = collapsed ? "펼치기" : "접기";
  });
  $("#output-root").addEventListener("input", () => syncRoot($("#output-root").value));
  $("#audio-output-root").addEventListener("input", () => syncRoot($("#audio-output-root").value));
  $("#new-job").addEventListener("click", newJob);
  $("#delete-data").addEventListener("click", previewDeletion);
  $("#refresh-delete").addEventListener("click", previewDeletion);
  $("#confirm-delete").addEventListener("click", deleteAllData);
  for (const id of ["close-delete", "cancel-delete"]) $("#" + id).addEventListener("click", () => { if (!state.deleting) $("#delete-dialog").close(); });
  $("#delete-dialog").addEventListener("cancel", (event) => { if (state.deleting) event.preventDefault(); });
  $("#job-select").addEventListener("change", selectHistory);
  $("#cancel-button").addEventListener("click", cancelGeneration);
  $("#tts-generate").addEventListener("click", () => startAudio());
  $("#retry-audio").addEventListener("click", () => startAudio());
  $("#tts-cancel").addEventListener("click", cancelAudio);
  $("#create-mapping").addEventListener("click", createMapping);
  $("#select-valid").addEventListener("click", () => bulkText(true));
  $("#clear-text").addEventListener("click", () => bulkText(false));
  $("#export-button").addEventListener("click", exportDataset);
  $("#copy-path").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(state.saveResult.output_dir); toast("저장 경로를 복사했습니다."); }
    catch (_) { toast("표시된 저장 경로를 선택해 복사해 주세요."); }
  });
}

// 초기 연결 실패 후에도 입력을 유지한 채 재연결한다.
async function connect() {
  try {
    const [config, preferences] = await Promise.all([api("/api/config"), api("/api/preferences")]);
    state.connected = true;
    $("#tts-model-label").textContent = config.tts_model || "LOCAL TTS";
    state.audioAvailable = config.audio_available;
    state.audioError = config.audio_error || "";
    $("#model-label").textContent = config.model_name;
    state.preferences = preferences;
    if (!state.preferencesLoaded) {
      syncRoot(recalled("output-root") ?? preferences.output_root);
      state.preferencesLoaded = true;
    }
    const jobs = await loadHistory();
    if (jobs.length && !state.job) {
      $("#job-select").value = jobs[0].id;
      await selectHistory();
    }
    showError("#connection-notice", "");
    render();
  } catch (error) {
    state.connected = false;
    showError("#connection-notice", error.message + " 최신 기능을 사용하려면 앱을 재시작해 주세요.");
    renderControls();
    setTimeout(connect, 4000);
  }
}

// 이벤트는 한 번만 연결하고 초안과 로컬 작업 상태를 복원한다.
function initialize() {
  try { restoreTypeWeights(JSON.parse(recalled("type-weights"))); } catch (_) { restoreTypeWeights(null); }
  try { restoreMoneyDigitWeights(JSON.parse(recalled("money-digit-weights"))); } catch (_) { restoreMoneyDigitWeights(null); }
  $("#prompt").value = recalled("prompt") || "";
  saveDraft();
  bindEvents();
  render();
  connect();
}

initialize();
