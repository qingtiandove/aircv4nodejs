"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const ac = require("../lib/index");
function findAllTemplateTest() {
    const sourceImg = ac.imread('testdata/2s.png');
    const searchImg = ac.imread('testdata/2t.png');
    const resultList = ac.findAllTemplate(sourceImg, searchImg);
    if (resultList) {
        console.log(resultList);
    }
    else {
        console.log("没有相似结果");
    }
}
function findAllSiftTest() {
    const sourceImg = ac.imread('testdata/1s.png');
    const searchImg = ac.imread('testdata/2s.png');
    const resultList = ac.findAllSift(sourceImg, searchImg);
    if (resultList) {
        console.log(resultList);
    }
    else {
        console.log("没有相似结果");
    }
}
findAllTemplateTest();
findAllSiftTest();
