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

#### CSV analysis of Camera of Midterm Project in the Camera.csv file here
