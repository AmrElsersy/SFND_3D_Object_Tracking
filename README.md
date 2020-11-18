# SFND 3D Object Tracking

Tracking front car to detect accident time

### Object Detection
* Used Yolo version 3 to detect Cars in the scene 

### Tracking
* Used classical vision algorithms detecting Keypoints & descriptors 
* Performed Matching KNN algorithm to match keypoints from frame at time(t) and time(t-1)

### Time to Collision
* TTC is function of the percentage of the avarage matches distance between 2 frames
* Calculated the TTC of the Camera and also by the LIDAR way

![Tracking](ezgif.com-video-to-gif.gif)

## Yolo Weights 
- you should download yolov3 weights to run the code
```
cd dat/yolo
wget https://pjreddie.com/media/files/yolov3.weights
```

### Final Algorithm

<img src="images/course_code_structure.png" width="779" height="414" />

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.


# SFND 3D Object Tracking
## [Rubric](https://review.udacity.com/#!/rubrics/2550/view) Points
---
#### 1. Match 3D Objects

"matchBoundingBoxes" function, which takes as the previous and the current dataframes and build a map that combines each boundingBox ID of the match of the previous frame to its corresponding keypoint (matched) in the current frame

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    auto bboxes_prev = prevFrame.boundingBoxes;
    auto bboxes_curr = currFrame.boundingBoxes;
    

    // empty 2D array
    int scores[bboxes_prev.size()][bboxes_curr.size()] = {0};

    for (auto match : matches)
    {
        // get 2 Keypoints of the match
        cv::KeyPoint prevKeypoint = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKeypoint = currFrame.keypoints[match.trainIdx];

        // get their points
        cv::Point prevPoint = prevKeypoint.pt;
        cv::Point currPoint = currKeypoint.pt;


        // store bboxes that each keypoint (2 keypoints of the match) inside
        std::vector<int> prevKeypointBBoxsIndex, currKeypointBBoxsIndex;

        for (int i = 0; i < prevFrame.boundingBoxes.size(); i++)
            if (prevFrame.boundingBoxes[i].roi.contains(prevPoint))
                prevKeypointBBoxsIndex.push_back(i);

        for (int i = 0; i < currFrame.boundingBoxes.size(); i++)
            if (currFrame.boundingBoxes[i].roi.contains(currPoint))
                currKeypointBBoxsIndex.push_back(i);

        
        for (auto i : prevKeypointBBoxsIndex)
            for (auto j : currKeypointBBoxsIndex)
                scores[i][j] ++;
    }

    for (int i = 0; i < prevFrame.boundingBoxes.size(); i++)
    {
        int bestScore = 0;
        int bestBBox = 0;
        for (int j = 0; j < currFrame.boundingBoxes.size(); j++)
        {
            if (scores[i][j] > bestScore)
            {
                bestScore = scores[i][j];
                bestBBox = j;
            }
        }

        bbBestMatches[i] = bestBBox;
    }
}
```

#### 2. Compute Lidar-based TTC

Time to Collision based on LIDAR only, first i filter the pointcloud and just keep the points within the lanewidth, then by competing the avarage distance of the points for both prev and current dataframe we get the TTC

```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{

    // filter the PointCloud to just take the lidar points between +/- laneWidth
    float laneWidth = 4;
    
    // to store depth
    std::vector<double> xPrev, xCurr;

    // filtering
    for (auto point : lidarPointsPrev)
        if (abs(point.y) < laneWidth/2)
            xPrev.push_back(point.x);

    for (auto point : lidarPointsCurr)
        if (abs(point.y) < laneWidth/2)
            xCurr.push_back(point.x);

    double minXPrev = 0; 
    double minXCurr = 0;
    if (xPrev.size() > 0)
    {  
       for (auto x: xPrev)
            minXPrev += x;
       minXPrev = minXPrev / xPrev.size();
    }
    if (xCurr.size() > 0)
    {  
       for (auto x: xCurr)
           minXCurr += x;
       minXCurr = minXCurr / xCurr.size();
    }  

    double dT = 1 /frameRate;
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}
```

#### 3. Associate Keypoint Correspondences with Bounding Boxes

Associating keypoint correspondences to the bounding boxes that contains that keyponts.
we also neglect the matches that their distance is below a certian threshold as a function of the mean (neglect the matches that are far away from the mean distance)

```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{

    std::vector<double> matchesDistance;
    double mean = 0;

    // Get the matches within the Bounding box
    for (auto match : kptMatches)
    {
        cv::KeyPoint currKeypoint = kptsCurr[match.trainIdx];
        cv::Point currPoint = currKeypoint.pt;
        cv::KeyPoint prevKeypoint = kptsPrev[match.queryIdx];
        cv::Point prevPoint = prevKeypoint.pt;

        if (boundingBox.roi.contains(currPoint))
            // matchesDistance.push_back(norm(currPoint-prevPoint));
            matchesDistance.push_back(match.distance);
    }

    for (auto distance : matchesDistance)
        mean += distance;    
    // for safty
    if (matchesDistance.size() == 0 )
        return;
    mean /= matchesDistance.size();

    // 
    double threshold = 0.7 * mean;

    for (auto match : kptMatches)
    {
        cv::KeyPoint currKeypoint = kptsCurr[match.trainIdx];
        cv::Point currPoint = currKeypoint.pt;
        cv::KeyPoint prevKeypoint = kptsPrev[match.queryIdx];
        cv::Point prevPoint = prevKeypoint.pt;

        // skip outside the bbox
        if (!boundingBox.roi.contains(currPoint))
            continue;

        // check fo distance mean
        // if (cv::norm(currPoint - prevPoint) <= threshold)
        if (match.distance <= threshold)
        {
            // boundingBox.keypoints.push_back(currKeypoint);
            boundingBox.kptMatches.push_back(match);
        }
    }

}

```

#### 4. Compute Camera-based TTC

Time to Collision based on Camera using keypoints, descriptors and matching algorithms.

```c++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
       
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {  
            double minDist = 100.0; // min. required distance
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }  
    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];   // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}    
```

#### 5.  Performance Evaluation 1

Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

Some examples the LIDAR TTC was wrong, because of the outliers, so we need to filter the outlires

![Screenshot 2020-11-18 20:41:31](https://user-images.githubusercontent.com/35613645/99573534-e5cfa000-29de-11eb-8afe-bbc2f4fb0b4a.png)


#### 6. Performance Evaluation 2

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Some examples with wrong TTC estimate of the Camera because of some outliers. Need to filter outliers, and @ some combinations of keypoints (not the current one) sometimes TTC becomes negative due to median distance ratio

![Screenshot 2020-11-18 21:32:15](https://user-images.githubusercontent.com/35613645/99578469-917bee80-29e5-11eb-8503-0f7926ac5392.png)



The TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles are:
- SHITOMASI/BRISK         
- SHITOMASI/BRIEF            
- SHITOMASI/ORB           


|Approach no. | Lidar | FAST + ORB | FAST + BRIEF |SHITOMASI + BRIEF |
|:---:|:---:|:---:|:---:|:---:|
|1 |13.5|10.3 |10.8 | 13.2|
|2 | 10.4|10.1 |11.0 | 13.0|
|3 | 22.2|11.1 |15.2 | 8.5|
|4 |14.3 |12.2 |14.4 | 15.0|
|5 |10.6 | 17.8|18.0 | 12.8|
|6 |13.7 |13.7 |12.3 | 14.3|
|7 | 11.8|11.3 |12.2 | 15.3|
|8 |12.2 |11.3 |13.8 | 12.6|
|9 |13.2 |12.1 |12.2 | 11.9|
|10 |15.7 |13.0 |13.1 | 12.6|

CSV analysis of Camera of Midterm Project
Detector, Descriptor ,Matches, Time detectors (ms), Time descriptors (ms), Total time
SHITOMASI,BRISK,93,7.1526378,0.30875096,7.4613868
SHITOMASI,BRISK,86,6.9459362,0.2412074,7.1871436
SHITOMASI,BRISK,78,6.9189176,0.2464945,7.165417
SHITOMASI,BRISK,88,7.1061466,0.25023222,7.35637
SHITOMASI,BRISK,80,7.6911968,0.24792138,7.9391172
SHITOMASI,BRISK,77,7.181048,0.2384144,7.4194624
SHITOMASI,BRISK,83,8.8692058,0.2461172,9.115323
SHITOMASI,BRISK,84,6.6520342,0.2450392,6.8970734
SHITOMASI,BRISK,80,6.9912416,0.25141116,7.2426508
SHITOMASI,BRIEF,113,7.7430192,0.27143942,8.0144596
SHITOMASI,BRIEF,109,7.0727188,0.21121352,7.2839382
SHITOMASI,BRIEF,102,7.7527506,0.21845768,7.9712122
SHITOMASI,BRIEF,99,7.0306278,0.25510674,7.2857414
SHITOMASI,BRIEF,100,8.1529238,0.28133938,8.4342622
SHITOMASI,BRIEF,100,8.9081902,0.20429374,9.112481
SHITOMASI,BRIEF,98,8.5601922,0.20676726,8.7669526
SHITOMASI,BRIEF,107,7.6752228,0.24898076,7.9242016
SHITOMASI,BRIEF,98,6.8410664,0.22888096,7.0699454
SHITOMASI,ORB,104,6.9276396,0.27513206,7.2027746
SHITOMASI,ORB,100,7.2935324,0.21122038,7.5047518
SHITOMASI,ORB,97,9.93622,0.25703734,10.193274
SHITOMASI,ORB,100,6.8392534,0.22016288,7.0594202
SHITOMASI,ORB,101,7.4286646,0.24281558,7.6714792
SHITOMASI,ORB,95,8.0542574,0.25779586,8.3120562
SHITOMASI,ORB,96,8.4334096,0.2062263,8.6396408
SHITOMASI,ORB,102,7.6239296,0.20507186,7.8290044
SHITOMASI,ORB,95,7.6927844,0.21185444,7.9046408
SHITOMASI,FREAK,84,7.229068,0.33891732,7.5679814
SHITOMASI,FREAK,88,7.1619968,0.2480919,7.4100936
SHITOMASI,FREAK,84,7.1952384,0.23750496,7.4327414
SHITOMASI,FREAK,86,6.696144,0.26323682,6.9593818
SHITOMASI,FREAK,84,6.9539918,0.23363592,7.1876238
SHITOMASI,FREAK,78,7.1271872,0.22995504,7.3571442
SHITOMASI,FREAK,79,7.5356218,0.2464798,7.7821016
SHITOMASI,FREAK,84,6.9039922,0.24472854,7.1487276
SHITOMASI,FREAK,83,8.3523734,0.23625938,8.5886318
SHITOMASI,SIFT,110,9.5741982,0.3780203,9.952194
SHITOMASI,SIFT,107,10.819396,0.30143722,11.120844
SHITOMASI,SIFT,102,10.60556,0.2717344,10.877314
SHITOMASI,SIFT,101,10.710518,0.27121108,10.981684
SHITOMASI,SIFT,97,11.046364,0.26140128,11.30773
SHITOMASI,SIFT,99,10.705716,0.29243102,10.998148
SHITOMASI,SIFT,94,11.083996,0.28598164,11.36996
SHITOMASI,SIFT,104,11.059496,0.29613542,11.355554
SHITOMASI,SIFT,95,10.262462,0.26020666,10.522652
HARRIS,BRISK,12,8.5478834,0.15408246,8.7019688
HARRIS,BRISK,10,8.2920936,0.07993076,8.3720322
HARRIS,BRISK,14,8.6740388,0.08196916,8.756006
HARRIS,BRISK,16,9.3622046,0.08427902,9.4464846
HARRIS,BRISK,16,26.404728,0.12142984,26.52615
HARRIS,BRISK,17,8.1537176,0.13948536,8.2932108
HARRIS,BRISK,15,10.752756,0.07697606,10.829686
HARRIS,BRISK,22,9.2966622,0.1210839,9.417751
HARRIS,BRISK,21,14.502138,0.1318002,14.633948
HARRIS,BRIEF,14,8.4058716,0.12813892,8.5340066
HARRIS,BRIEF,11,8.1970336,0.04245458,8.2394872
HARRIS,BRIEF,16,8.5399356,0.04494966,8.5848882
HARRIS,BRIEF,21,10.035984,0.06771604,10.103702
HARRIS,BRIEF,23,28.193522,0.07840196,28.271922
HARRIS,BRIEF,27,7.6698524,0.08910454,7.758954
HARRIS,BRIEF,16,10.734724,0.04936848,10.784116
HARRIS,BRIEF,25,9.0016822,0.0734706,9.0751528
HARRIS,BRIEF,24,15.174614,0.09728166,15.271928
HARRIS,ORB,12,7.9834426,0.10641918,8.0898608
HARRIS,ORB,12,8.196965,0.0407974,8.2377624
HARRIS,ORB,15,8.7681776,0.05072774,8.8189024
HARRIS,ORB,19,8.771882,0.05455268,8.8264386
HARRIS,ORB,23,26.643456,0.0839419,26.727344
HARRIS,ORB,21,7.9426354,0.15663928,8.0992688
HARRIS,ORB,15,10.809694,0.0500437,10.859674
HARRIS,ORB,25,10.441802,0.0833441,10.525102
HARRIS,ORB,23,15.973216,0.07447216,16.047696
HARRIS,FREAK,13,8.1317068,0.16137268,8.2930736
HARRIS,FREAK,12,8.4679448,0.06541206,8.5333598
HARRIS,FREAK,15,8.4785484,0.0858725,8.564416
HARRIS,FREAK,16,9.3300508,0.0942025,9.4242582
HARRIS,FREAK,16,27.437158,0.11229918,27.549466
HARRIS,FREAK,21,9.845472,0.11338012,9.958858
HARRIS,FREAK,12,10.466498,0.09070488,10.557246
HARRIS,FREAK,21,10.888976,0.10957968,10.99854
HARRIS,FREAK,19,14.473326,0.08776782,14.561134
HARRIS,SIFT,14,10.974138,0.113876,11.088014
HARRIS,SIFT,11,11.8972,0.06799044,11.965212
HARRIS,SIFT,16,11.88103,0.06753082,11.948552
HARRIS,SIFT,20,13.196778,0.07639198,13.273218
HARRIS,SIFT,21,29.939882,0.10060778,30.040528
HARRIS,SIFT,23,11.71835,0.14030758,11.858686
HARRIS,SIFT,13,13.763512,0.06205458,13.825546
HARRIS,SIFT,24,12.247354,0.11007752,12.357506
HARRIS,SIFT,23,17.679494,0.13588092,17.815322
FAST,BRISK,251,1.4352198,1.0681412,2.503361
FAST,BRISK,238,1.4353668,1.1581836,2.5935504
FAST,BRISK,236,1.4368564,1.0013836,2.4382498
FAST,BRISK,234,1.5537606,0.97268234,2.5264498
FAST,BRISK,211,1.4769874,0.93600976,2.4129952
FAST,BRISK,246,1.4720678,1.0646524,2.5367202
FAST,BRISK,243,1.7373048,1.0059014,2.7432062
FAST,BRISK,238,1.4876302,0.94033744,2.4279696
FAST,BRISK,242,1.5731744,1.435651,3.0088352
FAST,BRIEF,314,1.4954408,1.0064306,2.5018714
FAST,BRIEF,325,1.4985572,0.92594418,2.4245004
FAST,BRIEF,293,1.6648632,1.0028536,2.6677168
FAST,BRIEF,324,1.6358454,1.1527152,2.7885606
FAST,BRIEF,270,1.4313292,1.0527846,2.4841138
FAST,BRIEF,320,1.8960354,1.1243148,3.0203502
FAST,BRIEF,318,1.6024862,0.90398434,2.5064676
FAST,BRIEF,309,1.5868846,1.036595,2.6234796
FAST,BRIEF,301,1.7787882,0.91033376,2.68912
FAST,ORB,301,1.629005,1.1402398,2.7692448
FAST,ORB,302,1.5350622,0.9581166,2.4931788
FAST,ORB,292,1.4785064,0.96887798,2.4473834
FAST,ORB,315,1.6163042,1.0494526,2.6657568
FAST,ORB,277,1.662864,0.8851017,2.5479706
FAST,ORB,309,1.4524188,1.0679942,2.5204228
FAST,ORB,317,1.5184708,1.103676,2.6221468
FAST,ORB,296,1.7470264,0.90945764,2.656486
FAST,ORB,305,1.5265558,0.9499434,2.4764992
FAST,FREAK,246,1.4098574,1.1893868,2.5992344
FAST,FREAK,242,1.443442,1.0067442,2.4501862
FAST,FREAK,228,1.5579648,0.97826834,2.5362302
FAST,FREAK,250,1.7193218,0.95487966,2.6741946
FAST,FREAK,226,1.4759682,1.153215,2.6291832
FAST,FREAK,260,1.551193,0.9992178,2.5504108
FAST,FREAK,246,1.5766926,0.9880752,2.5647678
FAST,FREAK,248,1.552712,0.98147,2.534182
FAST,FREAK,242,1.5822982,1.0215912,2.6038894
FAST,SIFT,310,1.4240968,1.471127,2.8952238
FAST,SIFT,319,1.5785448,1.2756856,2.8542304
FAST,SIFT,291,1.4333578,1.2936882,2.7270558
FAST,SIFT,305,1.5580138,1.2665716,2.8245854
FAST,SIFT,285,1.5088276,1.3108088,2.8196462
FAST,SIFT,319,1.6589832,1.417864,3.0768472
FAST,SIFT,309,1.4867874,1.2789294,2.7657266
FAST,SIFT,294,1.6374428,1.5699698,3.2074126
FAST,SIFT,295,1.6256338,1.3898458,3.0154796
BRISK,BRISK,168,161.05516,0.62138958,161.67648
BRISK,BRISK,172,161.22862,0.59457972,161.82348
BRISK,BRISK,154,164.76152,0.57259048,165.33384
BRISK,BRISK,172,158.6081,0.60536364,159.21276
BRISK,BRISK,171,163.06318,0.6195462,163.68254
BRISK,BRISK,184,162.02438,0.57739542,162.6016
BRISK,BRISK,170,159.45188,0.60010398,160.05262
BRISK,BRISK,168,163.15824,0.6061104,163.76388
BRISK,BRISK,180,164.63412,0.53257512,165.16724
BRISK,BRIEF,174,156.00816,0.58991884,156.59812
BRISK,BRIEF,201,162.30956,0.62679624,162.93578
BRISK,BRIEF,181,159.49892,0.56068544,160.05948
BRISK,BRIEF,175,157.9662,0.55913116,158.5248
BRISK,BRIEF,179,157.18514,0.57112244,157.75648
BRISK,BRIEF,191,158.36114,0.56320208,158.92366
BRISK,BRIEF,203,160.10554,0.55067082,160.6563
BRISK,BRIEF,185,158.35232,0.51641198,158.86878
BRISK,BRIEF,179,165.00848,0.5673661,165.57492
BRISK,ORB,159,157.48502,0.62323982,158.1083
BRISK,ORB,172,161.62552,0.54207132,162.16746
BRISK,ORB,155,157.43112,0.53440674,157.96522
BRISK,ORB,164,165.2133,0.58782948,165.8013
BRISK,ORB,157,159.02852,0.61607602,159.64494
BRISK,ORB,178,163.35522,0.5439784,163.89912
BRISK,ORB,164,157.70258,0.53851882,158.2406
BRISK,ORB,168,160.48186,0.62705594,161.10906
BRISK,ORB,169,162.41834,0.48044108,162.89854
BRISK,FREAK,157,156.5795,0.54194588,157.12144
BRISK,FREAK,173,158.30528,0.53672248,158.84232
BRISK,FREAK,152,162.76134,0.53914112,163.30034
BRISK,FREAK,170,159.23334,0.56251314,159.79586
BRISK,FREAK,158,161.15806,0.54980352,161.70784
BRISK,FREAK,179,158.2847,0.67716824,158.96188
BRISK,FREAK,166,162.10964,0.53446946,162.64374
BRISK,FREAK,174,164.45184,0.50222354,164.95458
BRISK,FREAK,165,164.89676,0.48230504,165.37892
BRISK,SIFT,178,174.293,0.77587874,175.06916
BRISK,SIFT,189,176.05112,0.72834286,176.77926
BRISK,SIFT,166,179.24592,0.69737192,179.94368
BRISK,SIFT,179,183.18944,0.81633216,184.00578
BRISK,SIFT,168,181.84488,0.7667961,182.61222
BRISK,SIFT,191,182.44954,0.7186389,183.16788
BRISK,SIFT,190,181.7165,0.82478368,182.54068
BRISK,SIFT,172,185.82858,0.67734856,186.50576
BRISK,SIFT,179,183.47756,0.75752726,184.2351
ORB,BRISK,72,5.6722204,0.2416778,5.9138982
ORB,BRISK,73,5.3181758,0.21529424,5.5334622
ORB,BRISK,77,5.6858914,0.21970424,5.9055976
ORB,BRISK,83,5.4748778,0.22731884,5.7021986
ORB,BRISK,77,5.8307942,0.23739912,6.0681992
ORB,BRISK,90,5.3769464,0.2459653,5.6229166
ORB,BRISK,88,5.782539,0.2580585,6.0405926
ORB,BRISK,86,5.874855,0.24939236,6.1242454
ORB,BRISK,89,5.403818,0.28964782,5.6934668
ORB,BRIEF,48,5.275683,0.300762,5.576445
ORB,BRIEF,42,5.2271142,0.20418104,5.4312972
ORB,BRIEF,44,5.6301196,0.19684574,5.8269624
ORB,BRIEF,58,5.2517416,0.21279034,5.464529
ORB,BRIEF,52,5.2276336,0.24320464,5.4708402
ORB,BRIEF,76,5.3905488,0.2590189,5.6495726
ORB,BRIEF,67,6.2735386,0.34289906,6.6164308
ORB,BRIEF,82,6.2181098,0.24021858,6.4583274
ORB,BRIEF,65,5.6096376,0.2510417,5.8606744
ORB,ORB,66,5.0295266,0.23815862,5.2676862
ORB,ORB,69,5.5280722,0.1876308,5.715703
ORB,ORB,71,5.3521622,0.23617412,5.5883422
ORB,ORB,82,5.8462292,0.19301884,6.03925
ORB,ORB,89,5.1040164,0.2309468,5.3349632
ORB,ORB,99,5.4752796,0.31875284,5.7940344
ORB,ORB,90,5.564342,0.24948448,5.8138206
ORB,ORB,91,5.4801992,0.24080658,5.7210048
ORB,ORB,91,5.942475,0.25713044,6.1996074
ORB,FREAK,41,5.0072022,0.21616546,5.2233706
ORB,FREAK,35,5.28122,0.18021318,5.4614322
ORB,FREAK,43,5.414353,0.15963416,5.5739852
ORB,FREAK,46,5.6055118,0.15587978,5.7613906
ORB,FREAK,43,5.251134,0.15334648,5.4044844
ORB,FREAK,50,5.6958972,0.20507774,5.900972
ORB,FREAK,51,5.2556126,0.17161662,5.4272302
ORB,FREAK,47,5.4474672,0.16717036,5.6146356
ORB,FREAK,55,5.6388906,0.16626288,5.8051476
ORB,SIFT,66,5.8122624,0.26675894,6.0790184
ORB,SIFT,77,5.3761918,0.23961,5.6158018
ORB,SIFT,76,5.537441,0.23614472,5.7735818
ORB,SIFT,77,5.9998344,0.23962568,6.239464
ORB,SIFT,80,5.787341,0.24666306,6.034007
ORB,SIFT,93,5.5047188,0.27636882,5.7810886
ORB,SIFT,93,5.5490148,0.27984978,5.8288636
ORB,SIFT,92,6.2086038,0.26961858,6.4782214
ORB,SIFT,92,6.1223638,0.27999776,6.4023596
AKAZE,BRISK,134,54.420772,0.3730811,54.793858
AKAZE,BRISK,123,54.314932,0.31134992,54.626278
AKAZE,BRISK,126,53.052986,0.2962197,53.34924
AKAZE,BRISK,126,53.955566,0.36354864,54.319146
AKAZE,BRISK,128,57.60832,0.30904496,57.917412
AKAZE,BRISK,129,53.493986,0.32877334,53.822776
AKAZE,BRISK,139,53.184698,0.36791552,53.55259
AKAZE,BRISK,143,54.45125,0.39005568,54.84129
AKAZE,BRISK,141,54.301114,0.3297945,54.630884
AKAZE,BRIEF,138,53.984182,0.32517576,54.309444
AKAZE,BRIEF,131,50.034684,0.29118152,50.325842
AKAZE,BRIEF,128,49.612598,0.33641048,49.949032
AKAZE,BRIEF,127,50.890322,0.30188998,51.19226
AKAZE,BRIEF,131,50.026942,0.29756818,50.324568
AKAZE,BRIEF,143,50.308496,0.29031814,50.598772
AKAZE,BRIEF,147,51.991058,0.30294838,52.293976
AKAZE,BRIEF,145,50.614746,0.31517388,50.929914
AKAZE,BRIEF,149,49.857108,0.3194947,50.176588
AKAZE,ORB,128,49.457268,0.31225446,49.769594
AKAZE,ORB,126,52.716944,0.2690198,52.985954
AKAZE,ORB,124,50.581622,0.29251432,50.874054
AKAZE,ORB,115,50.919134,0.26676776,51.18589
AKAZE,ORB,127,51.48822,0.28000266,51.768304
AKAZE,ORB,128,50.892184,0.30370494,51.195886
AKAZE,ORB,134,51.925986,0.3188234,52.24478
AKAZE,ORB,132,53.92352,0.30141762,54.224968
AKAZE,ORB,142,56.763462,0.30356676,57.067066
AKAZE,FREAK,123,50.308202,0.35132706,50.659532
AKAZE,FREAK,126,51.65139,0.30150092,51.952838
AKAZE,FREAK,124,50.955394,0.31235344,51.267818
AKAZE,FREAK,119,51.947742,0.30860298,52.256344
AKAZE,FREAK,120,52.416672,0.33518548,52.751832
AKAZE,FREAK,130,51.078188,0.32656442,51.404724
AKAZE,FREAK,141,49.722946,0.33726308,50.060164
AKAZE,FREAK,144,55.41949,0.34698174,55.766508
AKAZE,FREAK,135,54.674396,0.33808138,55.012496
AKAZE,AKAZE,135,49.816928,0.3351698,50.152088
AKAZE,AKAZE,135,52.681762,0.2973516,52.979094
AKAZE,AKAZE,130,51.898056,0.29628634,52.194408
AKAZE,AKAZE,124,50.64738,0.33407514,50.981462
AKAZE,AKAZE,126,51.10161,0.34575674,51.447354
AKAZE,AKAZE,143,49.662088,0.31197026,49.974022
AKAZE,AKAZE,144,49.919044,0.34601154,50.265082
AKAZE,AKAZE,148,50.861118,0.3444063,51.20549
AKAZE,AKAZE,147,50.081626,0.3443083,50.4259
AKAZE,SIFT,131,57.694364,0.4078711,58.10224
AKAZE,SIFT,131,56.320992,0.3552549,56.676242
AKAZE,SIFT,127,55.68654,0.34591844,56.03248
AKAZE,SIFT,133,53.052104,0.36250984,53.414704
AKAZE,SIFT,134,55.858236,0.35422786,56.212506
AKAZE,SIFT,144,55.27396,0.36018136,55.63411
AKAZE,SIFT,144,55.89822,0.37670612,56.274932
AKAZE,SIFT,151,54.963594,0.3616004,55.325214
AKAZE,SIFT,148,54.625592,0.38500476,55.010634
SIFT,BRISK,63,66.17891,0.30365006,66.482514
SIFT,BRISK,65,65.2827,0.31863524,65.601396
SIFT,BRISK,62,66.895192,0.26111218,67.156264
SIFT,BRISK,66,67.046112,0.2619246,67.307968
SIFT,BRISK,58,67.172042,0.25254796,67.424588
SIFT,BRISK,63,67.511318,0.2796675,67.790912
SIFT,BRISK,63,65.859038,0.26350828,66.12256
SIFT,BRISK,66,68.17468,0.30456342,68.479264
SIFT,BRISK,78,65.248302,0.3702685,65.618546
SIFT,BRIEF,84,60.201988,0.3238704,60.525878
SIFT,BRIEF,76,62.083392,0.24703644,62.330352
SIFT,BRIEF,75,63.89306,0.2431331,64.136198
SIFT,BRIEF,84,63.954604,0.25900224,64.213618
SIFT,BRIEF,68,64.056622,0.25436488,64.31103
SIFT,BRIEF,73,64.50556,0.28473018,64.79025
SIFT,BRIEF,74,65.04897,0.24568012,65.294656
SIFT,BRIEF,69,64.097194,0.3007522,64.397956
SIFT,BRIEF,86,65.125018,0.31080406,65.435776
SIFT,FREAK,64,60.47041,0.31550414,60.785872
SIFT,FREAK,71,61.586728,0.260876,61.847604
SIFT,FREAK,64,67.467806,0.26214804,67.729956
SIFT,FREAK,66,65.151576,0.26610038,65.417646
SIFT,FREAK,58,67.318748,0.27301624,67.591776
SIFT,FREAK,58,70.56098,0.27838468,70.8393
SIFT,FREAK,63,68.119702,0.27756148,68.397336
SIFT,FREAK,64,68.324718,0.31731616,68.642042
SIFT,FREAK,77,68.35353,0.27156094,68.625088
SIFT,SIFT,80,59.140452,0.31682616,59.457286
SIFT,SIFT,79,61.35241,0.27280554,61.625242
SIFT,SIFT,84,67.737698,0.2631888,68.000926
SIFT,SIFT,92,66.994564,0.29602468,67.290622
SIFT,SIFT,88,68.198396,0.30228884,68.500628
SIFT,SIFT,79,67.33727,0.31344614,67.650674
SIFT,SIFT,80,64.343076,0.33039622,64.673434
SIFT,SIFT,100,68.12176,0.32002684,68.441828
SIFT,SIFT,102,68.360684,0.32703482,68.68771

