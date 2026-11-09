"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// 실제 업로드 함수를 모의 파일 읽기·API와 연결해 요청과 선택 상태를 검증한다.
async function main() {
  const nodes = new Map();
  const context = vm.createContext({
    console, setTimeout, clearTimeout,
    document: {querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, {value: "/tmp/output", hidden: true, textContent: ""});
      return nodes.get(selector);
    }},
    FileReader: class {
      // 선택한 파일 바이트를 실제 FileReader와 같은 데이터 URL 형태로 반환한다.
      readAsDataURL(file) {
        if (file.failRead) return this.onerror();
        this.result = "data:application/octet-stream;base64," + Buffer.from("xlsx bytes").toString("base64");
        this.onload();
      }
    }
  });
  const source = fs.readFileSync(path.join(__dirname, "../asr_corpus/web_static/app.js"), "utf8").replace(/initialize\(\);\s*$/, "");
  vm.runInContext(source + `
    renderControls = () => {};
    render = () => {};
    rememberGenerationSelection = () => {};
    toast = () => {};
    loadHistory = async () => {};
    setJob = (job) => {state.job = job;};
    const requests = [];
    let rejectUpload = false;
    api = async (url, options) => {
      requests.push({url, payload: JSON.parse(options.body)});
      if (rejectUpload) throw new Error("엑셀 8행: 정답을 입력하세요.");
      return {id: state.job?.id || "new-job", status: "completed", examples: [...(state.job?.examples || []), {id: "added", accepted: true, spoken_text: "발화", normalized_text: "정답"}]};
    };
    state.connected = true;
    globalThis.testApi = {state, requests, uploadExcel, setReject(value) {rejectUpload = value;}};
  `, context);
  const {state, requests, uploadExcel, setReject} = context.testApi;
  const event = (file = {}) => ({target: {value: "selected.xlsx", files: [{name: "test.xlsx", size: 100, ...file}]}});
  await uploadExcel(event());
  assert.equal(requests[0].url, "/api/jobs/manual");
  assert.equal(requests[0].payload.filename, "test.xlsx");
  assert.equal(Buffer.from(requests[0].payload.excel_base64, "base64").toString(), "xlsx bytes");
  assert.equal(state.job.examples.length, 1);
  assert.ok(state.generationSelection.has("added"));
  console.log("새 작업 생성·파일 전송·새 항목 자동 선택 통과");

  state.job.examples[0].id = "existing";
  state.generationSelection = new Set(["existing"]);
  const appendEvent = event();
  await uploadExcel(appendEvent);
  assert.equal(requests.at(-1).url, "/api/jobs/new-job/examples");
  assert.equal(state.job.examples.length, 2);
  assert.ok(state.generationSelection.has("existing") && state.generationSelection.has("added"));
  assert.equal(appendEvent.target.value, "");
  console.log("기존 목록·선택 유지·파일 재선택 통과");

  const before = state.job;
  setReject(true);
  await uploadExcel(event());
  assert.equal(state.job, before);
  assert.equal(nodes.get("#text-alert").textContent, "엑셀 8행: 정답을 입력하세요.");
  assert.equal(state.busy, false);
  setReject(false);
  const count = requests.length;
  for (const file of [{name: "test.xls"}, {size: 0}, {size: 5 * 1024 * 1024 + 1}, {failRead: true}]) await uploadExcel(event(file));
  assert.equal(requests.length, count);
  state.busy = true;
  await uploadExcel(event());
  state.busy = false;
  state.job.status = "running";
  await uploadExcel(event());
  assert.equal(requests.length, count);
  console.log("업로드 오류 시 목록 보존·파일 검증·진행 중 요청 차단 통과");
}

main().catch((error) => {console.error(error); process.exitCode = 1;});
