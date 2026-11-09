"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

// 실제 앱 파일을 서버 없이 열어 생성 단계·중간 결과·반응형 다크 화면을 검증한다.
function main() {
  const browser = process.env.CHROME_HEADLESS_SHELL;
  if (!browser) throw new Error("CHROME_HEADLESS_SHELL에 기존 Chromium 실행 파일을 지정하세요.");
  const root = path.join(__dirname, "../asr_corpus/web_static");
  const css = fs.readFileSync(path.join(root, "styles.css"), "utf8");
  const script = fs.readFileSync(path.join(root, "app.js"), "utf8").replace(/initialize\(\);\s*$/, "");
  const test = `
    function check(condition, message) { if (!condition) throw new Error(message); }
    state.connected = true;

    const item = {id: "0", type: "money", context: "amount", unit: "원", template: "이번 달 이용 금액은 {number} 입니다.", value: "35,000원"};
    state.job = {id: "fixture", title: "금액 안내 발화", status: "running", settings: {count: 1, type_weights: {money: 1}},
      counts: {accepted: 0}, examples: [], message: "배치 1 · 생성한 숫자를 LLM이 한글로 읽고 있습니다.",
      generation_batches: [{index: 1, phase: "spoken", status: "running", examples: [item]}]};
    render();
    check(document.querySelector('.process-step[aria-current="step"]').textContent.includes("한글 읽기"), "한글 읽기 진행 단계");
    check(document.querySelector(".template-preview").textContent.includes("{number}"), "템플릿 자리표시자");
    check(document.querySelector(".number-value").textContent.includes("35,000원"), "중간 숫자 표시");
    check(document.querySelector(".spoken-value").textContent.includes("대기"), "아직 생성하지 않은 읽기");
    const original = document.querySelector(".process-batch");
    original.open = false;
    item.spoken = "삼만 오천 원";
    item.spoken_text = "이번 달 이용 금액은 삼만 오천 원 입니다.";
    item.normalized_text = "이번 달 이용 금액은 35,000원 입니다.";
    item.accepted = true;
    item.result_id = "result";
    state.job.generation_batches[0].phase = "completed";
    state.job.generation_batches[0].status = "completed";
    state.job.examples = [{...item, id: "result", protected_terms: [], needs_review: false, reason: "unique", quality_issue: null}];
    state.job.status = "completed";
    state.job.message = "채택 목표 1개를 채웠습니다. 발화와 정답을 검토하세요.";
    state.job.counts.accepted = 1;
    render();
    check(!document.querySelector(".process-batch").open, "조회 중 접기 상태 유지");
    document.querySelector(".process-batch").open = true;
    check(document.querySelectorAll(".process-step.is-done").length === 4, "네 단계 완료 표시");
    check(document.querySelector("#process-batches .spoken-value").textContent.includes("삼만 오천 원"), "LLM 읽기 표시");
    check(document.querySelector("#process-batches .number-value").textContent.includes("35,000원"), "상세 생성 숫자 표시");
    check(document.querySelector(".process-mapping").textContent.includes("35,000원"), "최종 정답 표시");
    check(document.querySelector("#process-quotas").textContent.includes("금액 1 / 1"), "유형별 목표 표시");
    check(JSON.stringify(processQuotas({money: 3, count: 1}, 10).map(({type, quota}) => [type, quota])) === JSON.stringify([["money",8],["count",2]]), "서버와 동일한 유형 배분");
    const safe = processExample({...item, template: "<img src=x onerror=alert(1)> {number}"}, 0);
    check(!safe.querySelector("img"), "템플릿을 HTML로 실행하지 않음");
    check(getComputedStyle(document.documentElement).colorScheme === "dark", "어두운 테마");
    const color = getComputedStyle(document.body).backgroundColor;
    check(color === "rgb(15, 17, 23)", "차콜 배경");
    const completed = state.job.generation_batches[0];
    state.job.generation_batches.push({index: 2, phase: "failed", failed_phase: "spoken", status: "failed", error: "읽기 오류", examples: [{...item, spoken: "삼 원"}]});
    render();
    check(document.querySelectorAll(".process-batch").length === 2, "이전 배치 보존");
    check(document.querySelector(".process-step.is-stopped").textContent.includes("한글 읽기"), "실패 단계 표시");
    check(document.querySelector("#process-batches .alert").textContent.includes("읽기 오류"), "오류 표시");
    state.job.generation_batches = [completed];
    render();
    document.querySelector(".process-batch").open = true;
    if (window.innerWidth < 500) check(document.documentElement.scrollWidth <= window.innerWidth, "모바일 가로 넘침 없음");
    document.body.dataset.testResult = "passed";
  `;
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "asr-process-ui-"));
  try {
    const fixture = path.join(directory, "fixture.html");
    let html = fs.readFileSync(path.join(root, "index.html"), "utf8");
    html = html.replace('<link rel="stylesheet" href="/styles.css"><script src="/app.js" defer></script>', "<style>" + css + "</style>");
    html = html.replace("</body>", "<script>" + script + "\ntry {" + test + "} catch (error) {document.body.dataset.testResult = 'failed'; document.body.dataset.testError = error.message;}<" + "/script></body>");
    fs.writeFileSync(fixture, html);
    for (const [width, height] of [[1440, 1400], [390, 1200]]) {
      const screenshot = process.env.PROCESS_SCREENSHOT_DIR ? path.join(process.env.PROCESS_SCREENSHOT_DIR, "process-" + width + ".png") : null;
      const args = ["--no-sandbox", "--disable-gpu", "--disable-background-networking", "--no-first-run", "--hide-scrollbars", "--run-all-compositor-stages-before-draw", "--window-size=" + width + "," + height, "--virtual-time-budget=1000", "--dump-dom"];
      if (screenshot) args.push("--screenshot=" + screenshot);
      args.push("file://" + fixture);
      const output = execFileSync(browser, args, {encoding: "utf8", timeout: 30000, maxBuffer: 5 * 1024 * 1024, stdio: ["ignore", "pipe", "pipe"]});
      assert.match(output, /data-test-result="passed"/, output.match(/data-test-error="([^"]+)"/)?.[1] || "브라우저 검증 결과 없음");
      console.log(width + "px 생성 단계·중간 결과·히스토리·다크 테마 검증 통과");
    }
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
}

main();
