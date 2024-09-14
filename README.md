A simple solution using Foundation Models for the ood segmentation/tracking challenge [Obstacle Sequence Challenge](https://rrow2024.github.io/challenge.html)

![image](https://github.com/user-attachments/assets/5a18b5c4-a91b-4896-b501-1b83a4de8c04)

The core idea is illustrated in the image above. 
Currently, only the SOS dataset was used for the evaluation. All the procedure is realized in the [evaluate-dataset](https://github.com/pedrohtg/sos-challenge-2024/blob/main/evaluate-dataset.ipynb) notebook

A more detailed step-by-step is as follows:

For all test images:
- The GroundDINO detects objects and the road (following the `TEXT_PROMPT`).
- The detections are filtered, to remove know-objects (if they are in the `TEXT_PROMPT`) and to remove possible duplicates (boxes with high intersections, only the most confident is kept), and lastly reorder the boxes to position 'road' boxes first on the list.
- The boxes are passed, along the image, as input to the SAM model that provides a segmentation for each box.
- These segmentations are then labeled (and have a ood score attached if they are objects), using refined masks (through morphological operations)
- For each object segmentation, its convex hull is computed and only objects with some intersection with the road mask are mantained.
- The remainder objects are then allotted a track id. Tracking through frames is done using a greedy algorithm that matches boxes from one frame with boxes with the highest intersection in the next frame.
- Then both the track_id and ood_score maps are saved
