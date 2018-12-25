import * as ac from '../lib/index';
import {Mat} from "opencv4nodejs";

function findAllTemplateTest() {

    const sourceImg:Mat=ac.imread('testdata/2s.png');
    const searchImg:Mat=ac.imread('testdata/2t.png');

    const resultList=ac.findAllTemplate(sourceImg,searchImg);
    if(resultList){
        console.log(resultList)
    }else{
        console.log("没有相似结果")
    }
}

function findAllSiftTest() {

    const sourceImg:Mat=ac.imread('testdata/1s.png');
    const searchImg:Mat=ac.imread('testdata/2s.png');

    const resultList=ac.findAllSift(sourceImg,searchImg);
    if(resultList){
        console.log(resultList)
    }else{
        console.log("没有相似结果")
    }
}

findAllTemplateTest();

findAllSiftTest();
