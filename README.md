# CP-CNN

This repository contains the code for our paper

The proposed algorithm is implemented based on the [deepstream-plugin](https://github.com/vat-nvidia/deepstream-plugin)

## Requirements
The code was testd on Jetson Xavier AGX (Jetpack 3.1)

## Dataset
We tested our algorithm using Berkeley Deep Drive (BDD) dataset.

## Inference
### compile
    cd apps/trt-yolo/build/ && cmake -D .. && make -j8 && sudo make install && cd ../../../
    
### options
 IF, NOT define Engine,
 
     --precision=kFLOAT/kHALF 
     --deviceType=kGPU/kDLA
    
When define Engine, only FP16 can be used (automatically).

### demo
    ./apps/trt-yolo/build/trt-yolo-app --flagfile=config/yolov3.txt --deviceType=kGPU --precision=kHALF --inference_type=demo
    
### test
    ./apps/trt-yolo/build/trt-yolo-app --flagfile=config/yolov3.txt --deviceType=kGPU --precision=kHALF 

## Demo
We demonstrated our algorithm on Jetson AGX Xavier Embedded Platform. (Korea_road_video)

![AGX_demo_AdobeExpress](https://user-images.githubusercontent.com/35592964/236998064-257dff44-5e29-4154-847f-855406ea2cdb.gif)
