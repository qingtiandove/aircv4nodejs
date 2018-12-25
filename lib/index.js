"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const cv = require("opencv4nodejs");
exports.cv = cv;
const opencv4nodejs_1 = require("opencv4nodejs");
function showImg(img, title) {
    /*
    * 显示一个图片*/
    cv.imshow(title, img);
    cv.waitKey(0);
    cv.destroyAllWindows();
}
exports.showImg = showImg;
function imread(fileName) {
    /*
    * 确保文件存在*/
    const img = cv.imread(fileName);
    if (!img) {
        throw new Error("文件读取错误");
    }
    return img;
}
exports.imread = imread;
function findAllTemplate(sourceImg, searchImg, threshold = 0.5, maxcnt = 0, rgb = false, brgemove = false) {
    /*
    * threshold  阈值，当相识度小于阈值的时候，忽略掉*/
    if (rgb) {
        // todo  此种模式，暂不处理，有冲突
    }
    else {
        const sourceGray = sourceImg.cvtColor(cv.COLOR_BGR2GRAY);
        const searchGray = searchImg.cvtColor(cv.COLOR_BGR2GRAY);
        if (brgemove) {
            /*
            * 边界提取，来实现背景去除的功能*/
            sourceGray.canny(100, 200);
            searchGray.canny(100, 200);
        }
        const result = sourceGray.matchTemplate(searchGray, cv.TM_CCOEFF_NORMED);
        const { cols: width, rows: height } = searchImg;
        let resultList = [];
        while (true) {
            const resultMatch = result.minMaxLoc();
            const { maxVal, maxLoc: topLeft } = resultMatch;
            if (maxVal < threshold) {
                break;
            }
            const middlePoint = { x: topLeft.x + width / 2, y: topLeft.y + height / 2 };
            let resultPart = {};
            resultPart.result = middlePoint;
            const { x, y } = topLeft;
            resultPart.rectangle = [topLeft, new opencv4nodejs_1.Point2(x, y + height),
                new opencv4nodejs_1.Point2(x + width, y), new opencv4nodejs_1.Point2(x + width, y + height)];
            resultPart.confidence = maxVal;
            resultList.push(resultPart);
            if (maxcnt && resultList.length >= maxcnt) {
                break;
            }
            const emptyMat = new opencv4nodejs_1.Mat(0, 0, cv.CV_8UC3);
            const loDiff = maxVal - threshold + 0.1;
            const upDiff = 1;
            /*
            * floodfill   已经找到的地方填充掉*/
            result.floodFill(topLeft, -1000, emptyMat, loDiff, upDiff, cv.FLOODFILL_FIXED_RANGE);
        }
        return resultList;
    }
}
exports.findAllTemplate = findAllTemplate;
function findTemplate(sourceImg, searchImg, threshold = 0.5, rgb = false, bgremove = false) {
    const resultList = findAllTemplate(sourceImg, searchImg, threshold, 1, rgb, bgremove);
    return resultList.length ? [resultList[0]] : [];
}
exports.findTemplate = findTemplate;
function findAllSift(sourceImg, searchImg, minMatchCount = 4, maxcnt = 0) {
    /*
    * SIFT 特征点匹配*/
    if (cv.xmodules.xfeatures2d) {
        // const flann=cv.matchFlannBased()
        const siftDector = new cv.SIFTDetector({ edgeThreshold: 100 });
        let sourceKeyPoints = siftDector.detect(sourceImg);
        const searchKeyPoints = siftDector.detect(searchImg);
        if (searchKeyPoints.length < minMatchCount || sourceKeyPoints.length < minMatchCount) {
            return [];
        }
        let sourceDescriptors = siftDector.compute(sourceImg, sourceKeyPoints);
        const searchDescriptors = siftDector.compute(searchImg, searchKeyPoints);
        const { cols: width, rows: height } = searchImg;
        const matData = [[[0, 0]], [[0, height - 1]], [[width - 1, height - 1]], [[width - 1, 0]]];
        const matFromArray = new opencv4nodejs_1.Mat(matData, cv.CV_32FC2);
        let resultList = [];
        while (true) {
            let matAsArray = sourceDescriptors.getDataAsArray();
            const matchList = cv.matchKnnFlannBased(searchDescriptors, sourceDescriptors, 2);
            const good = [];
            matchList.forEach((match, index) => {
                if (match[0].distance < 0.9 * match[1].distance) {
                    good.push(match[0]);
                }
            });
            if (good.length < minMatchCount)
                break;
            let searchPts = [], sourcePts = [];
            for (let i = 0; i < good.length; i++) {
                const idQuery = good[i].queryIdx;
                const idTrain = good[i].trainIdx;
                const ptKpSource = sourceKeyPoints[idTrain].point;
                const ptKpSearch = searchKeyPoints[idQuery].point;
                searchPts.push(ptKpSearch);
                sourcePts.push(ptKpSource);
            }
            const resHome = cv.findHomography(searchPts, sourcePts, cv.RANSAC, 5);
            const resTrans = matFromArray.perspectiveTransform(resHome);
            // @ts-ignore
            const [lt, br, lt2, br2] = [resTrans.at(0, 0), resTrans.at(1, 0), resTrans.at(2, 0), resTrans.at(3, 0)];
            const middlePoint = { x: (lt2.x + lt.x) / 2, y: (lt2.y + lt.y) / 2 };
            let resultPart = {};
            resultPart.result = middlePoint;
            const x = (br.x + lt.x) / 2 - width / 2;
            const y = (br.y + lt.y) / 2 - height / 2;
            const topLeft = new opencv4nodejs_1.Point2(x, y);
            resultPart.rectangle = [topLeft, new opencv4nodejs_1.Point2(x, y + height), new opencv4nodejs_1.Point2(x + width, y),
                new opencv4nodejs_1.Point2(x + width, y + height)];
            resultPart.confidence = good.length / matchList.length;
            resultList.push(resultPart);
            if (maxcnt && resultList.length > maxcnt) {
                break;
            }
            const tindexes = [];
            for (let i = 0; i < good.length; i++) {
                tindexes.push(good[i].trainIdx);
            }
            let newKeySource = [];
            sourceKeyPoints.map(function (t, index) {
                if (tindexes.indexOf(index) === -1) {
                    newKeySource.push(t);
                }
            });
            sourceKeyPoints = newKeySource;
            let newDesSource = [];
            matAsArray.map(function (t, index) {
                if (tindexes.indexOf(index) === -1) {
                    newDesSource.push(t);
                }
            });
            matAsArray = newDesSource;
            sourceDescriptors = new opencv4nodejs_1.Mat(matAsArray, cv.CV_32FC1);
        }
        return resultList;
    }
    else {
        // todo  错误处理
        throw new Error('找不到对应的算法库');
    }
}
exports.findAllSift = findAllSift;
function findSift(sourceImg, searchImg, minMatchCount = 4) {
    const resultList = findAllSift(sourceImg, searchImg, minMatchCount, 0);
    return resultList.length ? [resultList[0]] : [];
}
exports.findSift = findSift;
function coordinateSift(sourceImg, searchImg, postion) {
    // 寻找与提供坐标最近的，的符合阈值的。一个结果
    const { x, y } = postion;
    const resultList = findAllSift(sourceImg, searchImg);
    let distanceList;
    for (let i = 0; i < resultList.length; i++) {
        const resX = Math.round(resultList[i].result.x);
        const resY = Math.round(resultList[i].result.y);
        const distance = Math.round(Math.sqrt(Math.pow(Math.abs(resX - x), 2) + Math.pow(Math.abs(resY - y), 2)));
        distanceList.push(distance);
    }
    const minDistance = Math.min.apply(null, distanceList);
    const indexDistance = distanceList.indexOf(minDistance);
    return [resultList[indexDistance]];
}
exports.coordinateSift = coordinateSift;
function findAll(sourceImg, searchImg, maxcnt = 0) {
    let resultList = findAllTemplate(sourceImg, searchImg, 0.5, maxcnt);
    if (!resultList.length) {
        resultList = findAllSift(sourceImg, searchImg, 4, maxcnt);
    }
    if (!resultList) {
        resultList = [];
    }
    return resultList;
}
exports.findAll = findAll;
function find(sourceImg, searchImg) {
    const resultList = findAll(sourceImg, searchImg, 1);
    return resultList.length ? [resultList[0]] : [];
}
exports.find = find;
function brightNess(img) {
    const hsvImg = img.cvtColor(cv.COLOR_BGR2HSV);
    // todo  计算图像亮度
}
exports.brightNess = brightNess;
