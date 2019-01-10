#	Betapose: Estimating 6D Pose From Localizing Designated Surface Keypoints
Please refer to our paper for detailed explanation. Arxiv Link is [here](https://arxiv.org/abs/1812.01387).
In the following, `ROOT` refers to the folder containing this README file.
## Ⅰ. Installation
1. All the codes are tested in Python, CUDA 8.0 and CUDNN 5.1.
2. Install [pytorch 0.4.0](https://github.com/pytorch/pytorch) and other dependencies.
3. Download LineMod dataset [here](http://ptak.felk.cvut.cz/6DB/public/datasets/hinterstoisser/). Only folders called models and test are needed. Put them in `DATAROOT/models` and `DATAROOT/test` where `DATAROOT` can be any folder you'd like to place LineMod dataset.
## Ⅱ. Designate Keypoints
**You can skip this step since we have provided designated keypoints files in '$ROOT/1_keypoint_designator/assets/sifts/'**.
1. The related code is in `$ROOT/1_keypoint_designator/`.
	```bash
	$ cd ROOT/1_keypoint_designator/
    ```
 2. Place the input model file (e.g. `DATAROOT/models/obj_01.ply`) in `$./assets/models/` 
 3. Build the code and run it. Just type:
 	```bash
	$ sh build_and_run.sh
	```
	The output file is in `$./assets/sifts/`. It's a ply file storing the 3D coordinates of designated keypoints.
## Ⅲ. Annotate Keypoints
1. The related code is in `$ROOT/2_keypoint_annotator/`.
	```bash
	$ cd ROOT/2_keypoint_annotator/
	```
2. Run keypoint annotator on one object of LineMod.
	```bash
	$ python annotate_keypoint.py --obj_id 1 --total_kp_number 50 --output_base ROOT/3_6Dpose_estimator/data --sixd_base DATAROOT
	```
	Type the following to see the meaning of options.
	```bash
	$ python annotate_keypoint.py -h
	```	
3. The annotated keypoints are in file `annot_train.h5` and `annot_eval.h5`. The corresponding training images are in folders `train` and `eval`.
## Ⅳ. Training
### Train Object Detector YOLOv3
1. Relative files locate in `$ROOT/3_6Dpose_estimator/train_YOLO`.
	```bash
	$ cd ROOT/3_6Dpose_estimator/train_YOLO
	```
2. Build Darknet (YOLOv3).
	```bash
	$ make
	```	
3. Prepare data as [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)'s instructions. Refer to folder `./scripts` for more help.
4. Download pretrained darknet53 [here](http://pjreddie.com/media/files/darknet53.conv.74)
5. Run `train_single.sh` or `train_all.sh` to train the network. 
6. Put trained weights (e.g. 01.weights) in folder `$ROOT/3_6Dpose_estimator/models/yolo/`.
### Train Keypoint Detector (KPD)
1.  Relative code is in `$ROOT/3_6Dpose_estimator/train_KPD/`
	```bash
	$ cd ROOT/3_6Dpose_estimator/train_KPD
	```
3. Modify Line 19, 21, 39, 46 of file`./src/utils/dataset/coco.py` to previously annotated dataset. Examples are given in these lines.
2. **Train on Linemod dataset without DPG**. 
	```bash
	$ python src/train.py --trainBatch 28 --expID seq5_Nov_1_1 --optMethod adam
	```
3. **Train on Linemod dataset with DPG**.  Just add a `--addDPG` option. and load the model trained after in the second step.
	```bash
	$ python src/train.py --trainBatch 28 --expID seq5_dpg_Nov_1_1 --optMethod adam --loadModel ./exp/coco/seq5_Nov_1_1/model_100.pkl --addDPG
	```
5. (Optional) Visualize training process. Type
	```bash
	$ tensorboard --logdir ./
	```

## Ⅴ. Evaluate
1.  Move back to the root of pose estimator.
	```bash
	$ cd ROOT/3_6Dpose_estimator/
	``` 
2. Run the following command.

	```bash
	$ CUDA_VISIBLE_DEVICES=1 python3 betapose_evaluate.py --nClasses 50 --indir /01/eval --outdir examples/seq1 --sp --profile
	```
	The output json file containing predicted 6D poses will be in examples/seq1.
