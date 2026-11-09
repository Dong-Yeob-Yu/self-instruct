"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// 실제 파형 함수를 로딩하여 WAV 청크·진폭·손상 파일 처리를 검증한다.
const source = fs.readFileSync(path.join(__dirname, "../asr_corpus/web_static/app.js"), "utf8").replace(/initialize\(\);\s*$/, "");
const context = vm.createContext({});
vm.runInContext(source, context);
const peaks = vm.runInContext("wavPeaks", context);

// 홀수 크기 부가 청크와 스테레오 진폭을 가진 PCM16 자료를 만든다.
function fixture() {
  const buffer = Buffer.alloc(70);
  buffer.write("RIFF"); buffer.writeUInt32LE(62, 4); buffer.write("WAVEJUNK", 8);
  buffer.writeUInt32LE(1, 16); buffer[20] = 42;
  buffer.write("fmt ", 22); buffer.writeUInt32LE(16, 26);
  buffer.writeUInt16LE(1, 30); buffer.writeUInt16LE(2, 32);
  buffer.writeUInt32LE(16000, 34); buffer.writeUInt32LE(64000, 38);
  buffer.writeUInt16LE(4, 42); buffer.writeUInt16LE(16, 44);
  buffer.write("data", 46); buffer.writeUInt32LE(16, 50);
  [0, 16384, -32768, 0, 8192, -8192, 0, 0].forEach((value, index) => buffer.writeInt16LE(value, 54 + index * 2));
  return buffer;
}

// Buffer의 실제 범위만 ArrayBuffer로 전달한다.
function arrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}
const wav = fixture();
const result = peaks(arrayBuffer(wav), 4);
assert.deepEqual(Array.from(result.peaks), [0.5, 1, 0.25, 0]);
assert.equal(result.duration, 4 / 16000);
assert.throws(() => peaks(new ArrayBuffer(8)), /WAV/);
const invalid = Buffer.from(wav); invalid.write("NOPE", 8);
assert.throws(() => peaks(arrayBuffer(invalid)), /WAV/);
assert.throws(() => peaks(arrayBuffer(wav.subarray(0, 65))), /불완전/);
const badChunk = Buffer.from(wav); badChunk.writeUInt32LE(1000, 50);
assert.throws(() => peaks(arrayBuffer(badChunk)), /청크/);
const nonPcm = Buffer.from(wav); nonPcm.writeUInt16LE(3, 30);
assert.throws(() => peaks(arrayBuffer(nonPcm)), /PCM16/);
const badAlign = Buffer.from(wav); badAlign.writeUInt16LE(2, 42);
assert.throws(() => peaks(arrayBuffer(badAlign)), /PCM16/);
console.log("실제 PCM16 진폭·스테레오·홀수 청크·손상 WAV 처리 검증 통과");
