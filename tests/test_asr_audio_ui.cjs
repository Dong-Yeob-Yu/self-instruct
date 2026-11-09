"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

// 실제 음성 장치 없이 브라우저 재생을 확인할 PCM16 WAV를 만든다.
function wavFixture() {
  const samples = 16000;
  const buffer = Buffer.alloc(44 + samples * 2);
  buffer.write("RIFF", 0); buffer.writeUInt32LE(buffer.length - 8, 4);
  buffer.write("WAVEfmt ", 8); buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(16000, 24); buffer.writeUInt32LE(32000, 28);
  buffer.writeUInt16LE(2, 32); buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36); buffer.writeUInt32LE(samples * 2, 40);
  return buffer;
}

// 실제 서버에 접속하지 않고 목록 선택·재생·저장·삭제를 검증한다.
async function main() {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_HEADLESS_SHELL, headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const root = path.join(__dirname, "../asr_corpus/web_static");
    const id = "a".repeat(32);
    const rows = [
      { id: "b".repeat(32), spoken_text: "배송 주소를 확인해 주세요.", normalized_text: "배송 주소를 확인해 주세요.", accepted: true, needs_review: false, reason: "unique", audio: { status: "missing" } },
      { id: "c".repeat(32), spoken_text: "우유 두 개를 주문했습니다.", normalized_text: "우유 2개를 주문했습니다.", accepted: true, needs_review: false, reason: "unique", audio: { status: "missing" } },
      { id: "d".repeat(32), spoken_text: "오늘 기온은 영하 삼점오 도씨입니다.", normalized_text: "오늘 기온은 -3.5℃입니다.", accepted: true, needs_review: true, reason: "reading_review", review_note: "단위 발음 확인: ℃ → 도씨", audio: { status: "missing" } },
      { id: "e".repeat(32), spoken_text: "우유 세 개를 주문했습니다.", normalized_text: "우유 3개를 주문했습니다.", accepted: false, needs_review: false, reason: "structure_duplicate", audio: { status: "missing" } },
      { id: "9".repeat(32), spoken_text: "", normalized_text: "", accepted: false, needs_review: false, reason: "표현 연결을 확인할 수 없습니다.", candidate: { parts: [{ text: "발화에 123이 남았습니다." }], number_generation_issue: "일반 문장의 숫자는 별도 표현 조각으로 분리하세요." }, audio: { status: "missing" } }
    ];
    const job = { id, title: "온도 발화", status: "completed", message: "생성 완료", updated_at: "2026-10-29T00:00:00Z", settings: { prompt: "", common_prompt: "", output_root: "/tmp/ai-dataset/", count: 3, batch_size: 3, similarity_threshold: 0.95, near_duplicate_policy: "review", prompt_mode: true }, counts: { accepted: 3, review: 1, rejected: 2, total: 5 }, examples: rows, audio: { status: "idle", message: "준비", counts: { total: 2, ready: 0, failed: 0 } } };
    let preferences = { common_prompt: "자연스러운 한국어로 숫자를 정확히 유지하세요.", output_root: "/tmp/ai-dataset/", reading_rules: [{ unit: "일", role: "*", reading: "sino" }, { unit: "분", role: "people", reading: "native" }] };
    let created = false;
    const requests = [];
    const screenshots = process.env.ASR_UI_SCREENSHOTS ? fs.mkdtempSync(path.join(os.tmpdir(), "asr-ui-preview-")) : null;
    await page.route("**/*", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      assert.equal(url.hostname, "self-instruct.test", "외부·기존 서버 요청 금지");
      const pathname = url.pathname;
      const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
      if (["/", "/app.js", "/styles.css"].includes(pathname)) {
        const file = pathname === "/" ? "index.html" : pathname.slice(1);
        return route.fulfill({ body: fs.readFileSync(path.join(root, file)), contentType: file.endsWith("html") ? "text/html" : file.endsWith("js") ? "text/javascript" : "text/css" });
      }
      if (pathname === "/api/config") return json({ model_name: "Gemma 4 E4B · 로컬", audio_available: true, audio_error: "" });
      if (pathname === "/api/preferences") {
        if (request.method() === "PATCH") { preferences = request.postDataJSON(); requests.push({ action: "preferences", payload: preferences }); }
        return json(preferences);
      }
      if (pathname === "/api/data/deletion-preview") {
        requests.push({ action: "deletion-preview" });
        return json({ token: "confirmed-preview", jobs: created ? 1 : 0, examples: created ? rows.length : 0, wav_files: created ? 3 : 0, datasets: created ? 1 : 0, files: created ? 12 : 0, bytes: 1024, roots: ["/tmp/ai-dataset/custom", "/tmp/ai-dataset/final"], skipped: 0 });
      }
      if (pathname === "/api/data/delete-all") {
        const payload = request.postDataJSON();
        assert.deepEqual(payload, { token: "confirmed-preview", confirmed: true });
        requests.push({ action: "delete-all", payload });
        created = false;
        return json({ deleted_jobs: 1, deleted_examples: rows.length, deleted_wav_files: 3, deleted_datasets: 1, deleted_files: 12, skipped: 0 });
      }
      if (pathname === "/api/jobs" && request.method() === "POST") {
        job.settings = { ...job.settings, ...request.postDataJSON() };
        requests.push({ action: "generate", payload: request.postDataJSON() });
        created = true;
        return json(job, 202);
      }
      if (pathname === "/api/jobs") return json({ jobs: created ? [job] : [] });
      if (pathname === "/api/jobs/" + id) return json(job);
      if (pathname === "/api/jobs/" + id + "/selection" && request.method() === "PATCH") {
        const payload = request.postDataJSON();
        requests.push({ action: "selection", payload });
        for (const row of rows.filter((item) => payload.example_ids.includes(item.id))) Object.assign(row, { accepted: payload.accepted, needs_review: false, reason: "manual_review" });
        return json(job);
      }
      const edit = pathname.match(/\/examples\/([0-9a-f]{32})$/);
      if (edit && request.method() === "PATCH") {
        const row = rows.find((item) => item.id === edit[1]);
        Object.assign(row, request.postDataJSON(), { needs_review: false, reason: "manual_review" });
        return json(job);
      }
      if (pathname === "/api/jobs/" + id + "/audio") {
        const payload = request.postDataJSON();
        requests.push({ action: "audio", payload });
        for (const row of rows.filter((item) => payload.example_ids.includes(item.id))) row.audio = { status: "ready", fingerprint: "f".repeat(64), duration_seconds: 1, url: "/api/jobs/" + id + "/examples/" + row.id + "/audio" };
        job.audio = { ...job.audio, status: "completed", message: "음성 생성 완료", counts: { total: 2, ready: 2, failed: 0 } };
        return json(job, 202);
      }
      const review = pathname.match(/\/examples\/([0-9a-f]{32})\/audio-review$/);
      if (review) {
        const payload = request.postDataJSON();
        requests.push({ action: "audio-review", payload });
        rows.find((row) => row.id === review[1]).audio.included = payload.included;
        return json(job);
      }
      if (pathname.endsWith("/audio")) return route.fulfill({ contentType: "audio/wav", body: wavFixture() });
      if (pathname === "/api/jobs/" + id + "/mapping") {
        const payload = request.postDataJSON();
        requests.push({ action: "mapping", payload });
        const result = { count: payload.example_ids.length, output_root: payload.output_root, output_dir: payload.output_root + "/dataset-ui-test", training_ready: false };
        job.audio.last_mapping = result;
        return json(result, 201);
      }
      return route.fulfill({ status: 404, body: "" });
    });
    await page.goto("http://self-instruct.test/");
    await page.waitForFunction(() => !document.querySelector("#generate-button").disabled);
    assert.equal(await page.locator("#output-root").inputValue(), "/tmp/ai-dataset/");
    assert.equal(await page.locator("#setup-stage").isVisible(), true);
    if (screenshots) await page.screenshot({ path: path.join(screenshots, "01-setup.png"), fullPage: true });
    await page.locator('[data-example="temperature"]').click();
    assert.match(await page.locator("#prompt").inputValue(), /온도/);
    await page.locator("#audio-output-root").fill("/tmp/ai-dataset/custom");
    assert.equal(await page.locator("#common-prompt, #reading-rules, #threshold, #duplicate-policy").count(), 0);
    assert.equal(await page.locator("#batch-size").inputValue(), "3");
    for (const size of ["1", "3", "5", "10"]) await page.locator("#batch-size").selectOption(size);
    for (const checkbox of await page.locator("[data-number-type]").all()) await checkbox.uncheck();
    await page.locator('[data-number-type="temperature"]').check();
    await page.locator('[data-type-weight="temperature"]').fill("3");
    assert.equal(await page.locator('[data-type-weight="money"]').isDisabled(), true);
    await page.locator("#count").fill("3");
    await page.locator("#generate-button").click();
    await page.locator("[data-text-choice]").first().waitFor();
    assert.equal(await page.locator("#text-stage").isVisible(), true);
    assert.equal(await page.locator("#setup-stage").isVisible(), true);
    const generation = requests.find((item) => item.action === "generate").payload;
    assert.equal(generation.prompt_mode, true);
    assert.deepEqual(generation.type_weights, { temperature: 3 });
    assert.equal(generation.batch_size, 10);
    assert.equal(generation.common_prompt, undefined);
    assert.equal(generation.reading_rules, undefined);
    assert.equal(await page.locator("#selected-count").textContent(), "0");
    assert.equal(await page.locator('[data-text-choice="' + rows[2].id + '"]').isChecked(), false);
    assert.equal(await page.locator('[data-text-choice="' + rows[3].id + '"]').isChecked(), false);
    assert.equal(await page.locator("#select-valid").isDisabled(), false);
    assert.equal(await page.getByText("숫자·고유명사만 바뀌었거나 문장 형식이 반복되어 자동 제외했습니다.").isVisible(), true);
    assert.equal(await page.locator(".review-note").filter({ hasText: "일반 문장의 숫자는 별도 표현 조각으로 분리하세요." }).isVisible(), true);
    assert.equal(await page.locator('[data-text-choice="' + rows[4].id + '"]').isDisabled(), true);
    await page.locator(".failed-source summary").click();
    assert.match(await page.locator(".failed-source pre").textContent(), /발화에 123이 남았습니다/);
    await page.locator(".failed-source summary").click();
    if (screenshots) await page.screenshot({ path: path.join(screenshots, "02-text.png"), fullPage: true });
    await page.locator("#select-valid").click();
    await page.waitForFunction(() => document.querySelector("#selected-count").textContent === "2");
    assert.equal(requests.filter((item) => item.action === "selection").length, 0);
    await page.locator('[data-text-choice="' + rows[0].id + '"]').uncheck();
    const reviewRow = page.locator('[data-row-id="' + rows[2].id + '"]');
    await reviewRow.locator('.inline-edit-actions input[type="checkbox"]').check();
    await reviewRow.locator("[data-row-save]").click();
    await page.waitForFunction(() => !document.querySelectorAll("[data-text-choice]")[2].disabled);
    await page.locator('[data-text-choice="' + rows[2].id + '"]').check();
    await page.waitForFunction(() => document.querySelector("#selected-count").textContent === "2");
    await page.locator("#tts-generate").click();
    await page.locator("#text-list audio").first().waitFor();
    assert.equal(await page.locator("#text-stage").isVisible(), true);
    const audio = requests.find((item) => item.action === "audio").payload;
    assert.deepEqual(audio.example_ids, [rows[1].id, rows[2].id]);
    assert.equal(audio.output_root, "/tmp/ai-dataset/custom");
    assert.equal(await page.locator("[data-audio-choice], .wav-download").count(), 0);
    assert.equal(await page.locator("#create-mapping").isDisabled(), false);
    assert.equal(await page.locator("[data-tts-select]").count(), 0);
    if (screenshots) await page.screenshot({ path: path.join(screenshots, "03-audio.png"), fullPage: true });
    await page.locator("#text-list audio").first().evaluate(async (player) => { await player.play(); player.dataset.playbackMarker = "keep"; });
    await page.waitForFunction(() => document.querySelector("#audio-selection-label").textContent === "2개 저장 예정");
    assert.equal(await page.locator("#text-list audio").first().getAttribute("data-playback-marker"), "keep");
    await page.locator('[data-text-choice="' + rows[1].id + '"]').uncheck();
    await page.waitForFunction(() => document.querySelector("#audio-selection-label").textContent === "1개 저장 예정");
    await page.locator("#audio-output-root").fill("/tmp/ai-dataset/final");
    await page.locator("#create-mapping").click();
    await page.locator("#mapping-status").filter({ hasText: "데이터셋 완성" }).waitFor();
    const mapping = requests.find((item) => item.action === "mapping").payload;
    assert.deepEqual(mapping.example_ids, [rows[2].id]);
    assert.equal(mapping.include_all_ready, true);
    assert.equal(mapping.output_root, "/tmp/ai-dataset/final");
    assert.match(await page.locator("#mapping-path").textContent(), /final\/dataset-ui-test/);
    await page.reload();
    await page.locator('[data-row-id="' + rows[1].id + '"]').waitFor();
    assert.equal(await page.locator("[data-audio-choice], .wav-download").count(), 0);
    assert.match(await page.locator("#mapping-status").textContent(), /1개 데이터셋 완성/);
    assert.equal(await page.locator("#audio-output-root").inputValue(), "/tmp/ai-dataset/final");
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    if (screenshots) await page.screenshot({ path: path.join(screenshots, "04-mobile.png"), fullPage: true });
    await page.locator("#delete-data").click();
    await page.locator("#delete-summary").filter({ hasText: "전체 파일 12개" }).waitFor();
    assert.equal(await page.locator("#delete-dialog").isVisible(), true);
    assert.match(await page.locator(".deletion-preserved").textContent(), /모델 파일과 공통 설정은 유지/);
    await page.locator("#cancel-delete").click();
    assert.equal(requests.filter((item) => item.action === "delete-all").length, 0);
    assert.equal(await page.locator("#text-stage").isVisible(), true);
    await page.locator("#delete-data").click();
    await page.waitForFunction(() => !document.querySelector("#confirm-delete").disabled);
    await page.locator("#confirm-delete").click();
    await page.waitForFunction(() => !document.querySelector("#delete-dialog").open && !document.querySelector("#setup-stage").hidden);
    assert.equal(requests.filter((item) => item.action === "delete-all").length, 1);
    assert.equal(await page.locator("#job-select option").count(), 1);
    assert.equal(await page.locator("#text-list .utterance-row").count(), 0);
    assert.equal(await page.locator("#text-list audio").count(), 0);
    assert.equal(await page.locator("#save-result").isVisible(), false);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    await page.locator("#delete-data").click();
    await page.locator("#delete-summary").filter({ hasText: "삭제할 생성 파일이 없습니다" }).waitFor();
    assert.equal(await page.locator("#confirm-delete").isDisabled(), true);
    await page.locator("#cancel-delete").click();
    await page.locator("#new-job").click();
    assert.deepEqual(errors, []);
    console.log("생성 설정 간소화, 생성·검토·재생·저장 및 전체 삭제 UI 검증 통과");
    if (screenshots) console.log("screenshots=" + screenshots);
  } finally { await browser.close(); }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
