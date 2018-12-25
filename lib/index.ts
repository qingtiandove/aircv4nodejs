import * as cv from 'opencv4nodejs';
import {Mat, Point2,KeyPoint,DescriptorMatch,Vec2} from 'opencv4nodejs';

export {cv};
// todo 1.测试用例

// travis    gihub的构建，
interface ResultPart {
    result?:{x:number,y:number};
    rectangle?:Point2[];
    confidence?:number;
}
export function showImg(img:Mat,title:string) {
    /*
    * 显示一个图片*/
    cv.imshow(title,img);
    cv.waitKey(0);
    cv.destroyAllWindows();
}

export function imread(fileName:string):Mat {
    /*
    * 确保文件存在*/
    const img=cv.imread(fileName);
    if(!img){
        throw new Error("文件读取错误");
    }
    return img;
}

export function findAllTemplate(sourceImg:Mat,searchImg:Mat,threshold=0.5,
                         maxcnt=0,rgb=false,brgemove=false):ResultPart[] {
    /*
    * threshold  阈值，当相识度小于阈值的时候，忽略掉*/
    if(rgb){
        // todo  此种模式，暂不处理，有冲突
    }else{
        const sourceGray:Mat=sourceImg.cvtColor(cv.COLOR_BGR2GRAY);
        const searchGray:Mat=searchImg.cvtColor(cv.COLOR_BGR2GRAY);

        if(brgemove){
            /*
            * 边界提取，来实现背景去除的功能*/
            sourceGray.canny(100,200);
            searchGray.canny(100,200);
        }
        const result:Mat=sourceGray.matchTemplate(searchGray,cv.TM_CCOEFF_NORMED);
        const {cols:width,rows:height}=searchImg;
        let resultList:ResultPart[]=[];
        while (true){
            const resultMatch=result.minMaxLoc();
            const { maxVal,  maxLoc:topLeft }=resultMatch;
            if(maxVal<threshold){
                break;
            }
            const middlePoint:{x:number,y:number}={x:topLeft.x+width/2,y:topLeft.y+height/2};
            let resultPart:ResultPart={};
            resultPart.result=middlePoint;
            const {x,y}=topLeft;
            resultPart.rectangle=[topLeft,new Point2(x,y+height),
                new Point2(x+width,y),new Point2(x+width,y+height)];
            resultPart.confidence=maxVal;
            resultList.push(resultPart);
            if(maxcnt && resultList.length>=maxcnt){
                break;
            }

            const emptyMat:Mat=new Mat(0,0,cv.CV_8UC3);
            const loDiff:number=maxVal-threshold+0.1;
            const upDiff:number=1;
            /*
            * floodfill   已经找到的地方填充掉*/
            result.floodFill(topLeft,-1000,emptyMat,loDiff,upDiff,cv.FLOODFILL_FIXED_RANGE);
        }
        return resultList;
    }
}

export function findTemplate(sourceImg:Mat,searchImg:Mat,threshold=0.5, rgb=false, bgremove=false) {
    const resultList=findAllTemplate(sourceImg,searchImg,threshold,1,rgb,bgremove);
    return resultList.length?[resultList[0]]:[];
}

export function findAllSift(sourceImg:Mat,searchImg:Mat,minMatchCount=4,maxcnt=0) {
    /*
    * SIFT 特征点匹配*/
    if(cv.xmodules.xfeatures2d){
        // const flann=cv.matchFlannBased()

        const siftDector=new cv.SIFTDetector({edgeThreshold:100});

        let sourceKeyPoints:KeyPoint[]=siftDector.detect(sourceImg);
        const searchKeyPoints:KeyPoint[]=siftDector.detect(searchImg);

        if(searchKeyPoints.length<minMatchCount || sourceKeyPoints.length<minMatchCount){
            return [];
        }
        let sourceDescriptors=siftDector.compute(sourceImg,sourceKeyPoints);
        const searchDescriptors=siftDector.compute(searchImg,searchKeyPoints);

        const {cols:width,rows:height}=searchImg;

        const matData= [[[0,0]], [[0,height-1]], [[width-1, height-1]],[[width-1, 0]]];

        const matFromArray=new Mat(matData,cv.CV_32FC2);
        let resultList:ResultPart[]=[];
        while (true){
            let matAsArray=sourceDescriptors.getDataAsArray();

            const matchList:DescriptorMatch[][]=cv.matchKnnFlannBased(searchDescriptors,sourceDescriptors,2);

            const good:DescriptorMatch[]=[];
            matchList.forEach((match,index)=>{
                if(match[0].distance<0.9*match[1].distance){
                    good.push(match[0]);
                }
            });
            if(good.length<minMatchCount) break;
            let searchPts:Point2[]=[],sourcePts:Point2[]=[];

            for (let i = 0; i <good.length ; i++) {
                const idQuery=good[i].queryIdx;
                const idTrain=good[i].trainIdx;

                const ptKpSource=sourceKeyPoints[idTrain].point;
                const ptKpSearch=searchKeyPoints[idQuery].point;

                searchPts.push(ptKpSearch);
                sourcePts.push(ptKpSource);
            }

            const resHome=cv.findHomography(searchPts,sourcePts,cv.RANSAC,5);

            const resTrans=matFromArray.perspectiveTransform(resHome);

            // @ts-ignore
            const [lt, br,lt2,br2]:Vec2[] = [resTrans.at(0,0),resTrans.at(1,0),resTrans.at(2,0),resTrans.at(3,0)];

            const middlePoint={x:(lt2.x+lt.x)/2,y:(lt2.y+lt.y)/2};
            let resultPart:ResultPart={};

            resultPart.result=middlePoint;

            const x=(br.x+lt.x)/2-width/2;
            const y=(br.y+lt.y)/2-height/2;

            const topLeft=new Point2(x,y);
            resultPart.rectangle=[topLeft,new Point2(x, y+height),new Point2(x+width,y ),
                new Point2(x+width, y+height)];
            resultPart.confidence=good.length/matchList.length;

            resultList.push(resultPart);
            if(maxcnt && resultList.length>maxcnt){
                break;
            }
            const tindexes:number[]=[];
            for (let i = 0; i <good.length ; i++) {
                tindexes.push(good[i].trainIdx);
            }

            let newKeySource:any[]=[];
            sourceKeyPoints.map(function (t,index) {
                if(tindexes.indexOf(index)===-1){
                    newKeySource.push(t);
                }
            });
            sourceKeyPoints=newKeySource;

            let newDesSource:any[]=[];
            matAsArray.map(function (t,index) {
                if(tindexes.indexOf(index)===-1){
                    newDesSource.push(t);
                }
            });
            matAsArray=newDesSource;
            sourceDescriptors=new Mat(matAsArray,cv.CV_32FC1);
        }
        return resultList;
    }else{
        // todo  错误处理
        throw new Error('找不到对应的算法库')
    }
}

export function findSift(sourceImg:Mat,searchImg:Mat,minMatchCount=4) {
    const resultList=findAllSift(sourceImg,searchImg,minMatchCount,0)

    return resultList.length ?[resultList[0]]:[];
}

export function coordinateSift(sourceImg:Mat,searchImg:Mat,postion:{x:number,y:number}){
    // 寻找与提供坐标最近的，的符合阈值的。一个结果
    const {x,y}=postion;

    const resultList:ResultPart[]=findAllSift(sourceImg,searchImg);

    let distanceList:number[];
    for (let i = 0; i < resultList.length; i++) {
        const resX=Math.round(resultList[i].result.x);
        const resY=Math.round(resultList[i].result.y);

        const distance=Math.round(Math.sqrt(Math.pow(Math.abs(resX-x),2)+Math.pow(Math.abs(resY-y),2)));
        distanceList.push(distance);
    }

    const minDistance=Math.min.apply(null,distanceList);
    const indexDistance=distanceList.indexOf(minDistance);
    return [resultList[indexDistance]]
}

export function findAll(sourceImg:Mat,searchImg:Mat,maxcnt=0):ResultPart[] {
    let resultList=findAllTemplate(sourceImg,searchImg,0.5,maxcnt);
    if(!resultList.length){
        resultList=findAllSift(sourceImg,searchImg,4,maxcnt);
    }
    if(!resultList){
        resultList=[];
    }
    return resultList;
}

export function find(sourceImg:Mat,searchImg:Mat) {
    const resultList=findAll(sourceImg,searchImg,1);

    return resultList.length?[resultList[0]]:[];
}

export function brightNess(img:Mat) {
    const hsvImg:Mat=img.cvtColor(cv.COLOR_BGR2HSV);
    // todo  计算图像亮度
}
