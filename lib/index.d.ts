import * as cv from 'opencv4nodejs';
import { Mat, Point2 } from 'opencv4nodejs';
export { cv };
interface ResultPart {
    result?: {
        x: number;
        y: number;
    };
    rectangle?: Point2[];
    confidence?: number;
}
export declare function showImg(img: Mat, title: string): void;
export declare function imread(fileName: string): Mat;
export declare function findAllTemplate(sourceImg: Mat, searchImg: Mat, threshold?: number, maxcnt?: number, rgb?: boolean, brgemove?: boolean): ResultPart[];
export declare function findTemplate(sourceImg: Mat, searchImg: Mat, threshold?: number, rgb?: boolean, bgremove?: boolean): ResultPart[];
export declare function findAllSift(sourceImg: Mat, searchImg: Mat, minMatchCount?: number, maxcnt?: number): ResultPart[];
export declare function findSift(sourceImg: Mat, searchImg: Mat, minMatchCount?: number): ResultPart[];
export declare function coordinateSift(sourceImg: Mat, searchImg: Mat, postion: {
    x: number;
    y: number;
}): ResultPart[];
export declare function findAll(sourceImg: Mat, searchImg: Mat, maxcnt?: number): ResultPart[];
export declare function find(sourceImg: Mat, searchImg: Mat): ResultPart[];
export declare function brightNess(img: Mat): void;
