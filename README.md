# Mask-RCNN
Practice implementation on Mask R-CNN.

### Current best (After grid search)  
Used settings:  
rpn_pos_weight: 13.7  
roi_pos_weight: 2.0
rpn_nms_threshs: 0.6  
rpn_nms_topk_trains: 2000  
rpn_nms_topk_tests: 500  
rpn_nms_topk_posts: 500  
roi_test_score_threshs: 0.5  
roi_nms_threshs: 0.4  
roi_nms_topk_posts: 10  
  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.719  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.333  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.191  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.468  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.261  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.444  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616  
mAP: 0.4286285416503737, mAR: 0.5051209714233543  
