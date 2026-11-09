"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// 브라우저와 서버 없이 실제 UI 함수의 유형 선택·복원·요청 payload를 검사한다.
async function main() {
  const html = fs.readFileSync(path.join(__dirname, "../asr_corpus/web_static/index.html"), "utf8");
  const types = [...html.matchAll(/data-number-type="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(types.length, 10);
  assert.equal(new Set(types).size, 10);
  const checkboxes = types.map((type) => ({ dataset: { numberType: type }, checked: true }));
  const inputs = Object.fromEntries(types.map((type) => [type, { value: "1", disabled: false }]));
  const nodes = { "#prompt": { value: "상담 안내" }, "#output-root": { value: "/tmp/test" },
    "#count": { value: "8" }, "#batch-size": { value: "3" }, "#generation-form": { reportValidity: () => true } };
  const digitInputs = Array.from({length: 9}, (_, index) => ({dataset: {moneyDigit: String(index + 1)}, value: "1"}));
  const requests = [];
  const errors = [];
  const storage = new Map();
  const context = vm.createContext({ assert, requests, errors, console, setTimeout, clearTimeout,
    window: { scrollTo() {} }, localStorage: { setItem: (key, value) => storage.set(key, value) },
    document: {
      querySelectorAll: (selector) => selector === "[data-number-type]" ? checkboxes : selector === "[data-money-digit]" ? digitInputs : [],
      querySelector: (selector) => {
        const match = selector.match(/^\[data-type-weight="([^"]+)"\]$/);
        return match ? inputs[match[1]] : nodes[selector];
      },
    },
  });
  const source = fs.readFileSync(path.join(__dirname, "../asr_corpus/web_static/app.js"), "utf8").replace(/initialize\(\);\s*$/, "");
  vm.runInContext(source, context);
  await vm.runInContext(`(async () => {
    restoreMoneyDigitWeights({"2": 1, "9": 3});
    assert.equal(JSON.stringify(selectedMoneyDigitWeights()), JSON.stringify({"2": 1, "9": 3}));
    restoreTypeWeights({money: 3, count: 1});
    assert.equal(JSON.stringify(selectedTypeWeights()), JSON.stringify({money: 3, count: 1}));
    assert.equal(document.querySelector('[data-type-weight="phone"]').disabled, true);
    renderControls = () => {};
    render = () => {};
    setJob = () => {};
    loadHistory = async () => {};
    showError = (selector, message) => { if (message) errors.push(message); };
    api = async (url, options) => { requests.push(JSON.parse(options.body)); return {id: "test"}; };
    await startGeneration({preventDefault() {}});
    assert.equal(requests.length, 1);
    assert.equal(JSON.stringify(requests[0].type_weights), JSON.stringify({money: 3, count: 1}));
    assert.equal(requests[0].count, 8);
    assert.equal(requests[0].auto_audio, false);
    assert.equal(JSON.stringify(requests[0].money_digit_weights), JSON.stringify({"2": 1, "9": 3}));
    restoreMoneyDigitWeights({});
    await startGeneration({preventDefault() {}});
    assert.equal(requests.length, 1);
    assert.match(errors.pop(), /자릿수 비율/);
    restoreMoneyDigitWeights(null);
    assert.equal(Object.keys(selectedMoneyDigitWeights()).length, 9);
    restoreTypeWeights({});
    await startGeneration({preventDefault() {}});
    assert.equal(requests.length, 1);
    assert.match(errors[0], /유형을 하나 이상/);
    restoreTypeWeights(null);
    assert.equal(Object.keys(selectedTypeWeights()).length, 10);
  })()`, context);
  console.log("유형 선택·비율 복원·생성 payload·미선택 차단 검증 통과");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
