#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tesseract/baseapi.h>

#include <iostream>
#include <vector>

#include <string.h>



// hough() trackbar params
double dp = 1; int dpTbar = 1;
double minDist = 0; int minDistTbar = 100/8;
int param1 = 100;
int param2 = 50;
int minRadius = 10;
int maxRadius = 50;

int dpTbarPrev = 0;
int minDistTbarPrev = 0;
int param1Prev = 0;
int param2Prev = 0;
int minRadiusPrev = 0;
int maxRadiusPrev = 0;

// contourFilter() trackbar params
int minContourThresh = 15;
int maxContourThresh = 280;
int dilateSize = 3;
int minContourThreshPrev = 0;
int maxContourThreshPrev = 0;
int dilateSizePrev = 0;


void contourDetect(cv::Mat maskedVector);
bool print, singleImage, save, video;
std::string file,ext,imageFile;
int frameNumber;

cv::Mat getSkel(cv::Mat &img);


void parseFilename(char *filename)
{
    char *token;
    char delimiter[] = ".";
    int j=0;
    token = strtok(filename, delimiter);                
    while(token != NULL)
    {       
        if(j==0)
            file = std::string(filename);
        else if(j==1)
            ext = std::string(token);
        token = strtok(NULL, delimiter);
        j++;
    }
}

std::string ocr(cv::Mat &img)
{
    tesseract::TessBaseAPI tess;
    tess.Init(0, "spa", tesseract::OEM_DEFAULT);
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tess.SetVariable("tessedit_char_whitelist", "0123456789"); //To find only digits
    tess.SetImage((uchar*)img.data, img.cols, img.rows, 1, img.step1());
    std::string text = tess.GetUTF8Text();
    return text;
        
    /*

    tesseract::ResultIterator* ri = tess.GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (ri != 0) {
    do {
      std::cout << ri->GetUTF8Text(level) << "\t"<< ri->Confidence(level) <<"\n";
    } while (ri->Next(level));
    }

    */
}



void draw(cv::Mat &image, std::vector<cv::Vec3f> &circles)
{        
    cv::Mat drawn = image.clone();
    
    if(circles.size() == 0)
        return ;
    
    //Draw detected circles in image
    std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
    while(itc != circles.end())
    {
        circle(drawn, cv::Point((*itc)[0],(*itc)[1]), (*itc)[2], cv::Scalar(102,255,0), 8, 8, 0);
        itc++;
    }
    
    if(!video)
    {
        cv::imshow("Circles", drawn);
        if(save)
            cv::imwrite(file+std::string("_circles_")+"."+ext,drawn);
    }
    else if(video)
    {    
        // Resize images to display, when using videos
        cv::resize(drawn, drawn, cv::Size(drawn.cols/2, drawn.rows/2));
        cv::Mat aux = image.clone();
        cv::resize(aux, aux, cv::Size(aux.cols/2, aux.rows/2));

        cv::imshow("Circles", drawn);
        if(save)
        {
            cv::imwrite(std::string("video1/video_circles_")+std::to_string(frameNumber)+std::string(".jpg"),drawn);
            cv::imwrite(std::string("video1/video_original_")+std::to_string(frameNumber)+std::string(".jpg"),aux);
        }
    }
}


bool houghTrackbar()
{
    if(!print)
        return true;
    if((dpTbar != dpTbarPrev)||(minDistTbar!=minDistTbarPrev)||(param1!=param1Prev)||(param2!=param2Prev)||(minRadius!=minRadiusPrev)||(maxRadius!=maxRadiusPrev))
    {
        dpTbarPrev = dpTbar;
        minDistTbarPrev = minDistTbar;
        param1Prev = param1;
        param2Prev = param2;
        minRadiusPrev = minRadius;
        maxRadiusPrev = maxRadius;
        return true;
    }
    return false;
}


/**
    Searches for circles in image using hough transform
*/
std::vector<cv::Vec3f> hough(cv::Mat &image)
{
    std::vector<cv::Vec3f> circles;
    cv::Mat grey = image.clone();
    cvtColor(image, grey, CV_BGR2GRAY);

    dp = (double) dpTbar/100;        
    minDist = (double) minDistTbar/100*grey.rows;
    
    cv::HoughCircles(grey,circles,CV_HOUGH_GRADIENT,
        dp,    // accumulator resolution
        minDist, // minimum distance between two circles
        param1,   // canny high threshold 200
        param2,   // minimum number of votes 100
        minRadius, maxRadius);  // min and max radius 0,0
    

    std::cout << file << " - found circles : " << circles.size() << "\n";    
    
    draw(image,circles);    
    
    if(circles.size() == 0)
        return {};
    
    return circles;
}


/**
    Returns cropped images, with circles found using hough()
*/
std::vector<cv::Mat> crop(cv::Mat &image, std::vector<cv::Vec3f> &circles)
{
    if((circles.size() == 0))
     return {};

    cv::Mat grey;
    cvtColor(image, grey, CV_BGR2GRAY);
    std::vector<cv::Mat> maskedVector;

    std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
    int i = 0;

    while(itc != circles.end())
    {
        //ROI
        float vertFactor = (float) 2/3;
        float horFactor = (float) 5/6;

        cv::Rect roi((*itc)[0]-(*itc)[2], (*itc)[1]-(*itc)[2],
            (*itc)[2]*2, (*itc)[2]*2);
       
        cv::Mat crop = cv::Mat::zeros(roi.size(), CV_8UC1);;
        crop = grey(roi);
        cv::Mat cropAux = crop.clone();

        cv::Mat maskedVectorImg = crop(
                cv::Rect(cv::Point(crop.cols*(1-horFactor)/2,crop.rows*(1-vertFactor)/2), 
                cv::Size(crop.cols*horFactor,crop.rows*vertFactor)));
        cv::equalizeHist(maskedVectorImg,maskedVectorImg);
        cv::bitwise_not(maskedVectorImg,maskedVectorImg);
        maskedVectorImg = maskedVectorImg > 150;
           
        if(save)
        {
            cv::imwrite(file+std::string("_crop_")+std::to_string(i)+"."+ext,cropAux);
            cv::imwrite(file+std::string("_roi_")+std::to_string(i)+"."+ext,maskedVectorImg);
        }
        

        maskedVector.push_back(maskedVectorImg);
        itc++;i++;
    }
    
    if(print)
        cv::imshow("ROI", maskedVector[0]);
    
    return maskedVector;
}


bool contourTrackbar()
{
    if(!print)
        return true;
    
    if(dilateSize <= 1)
        dilateSize = 1;
        
    if((minContourThresh != minContourThreshPrev)||(dilateSize != dilateSizePrev)||(maxContourThresh != maxContourThreshPrev))
    {
        minContourThreshPrev = minContourThresh;
        maxContourThreshPrev = maxContourThresh;
        dilateSizePrev = dilateSize;
        return true;
    }
    return false;
}


/**
    Checks if any point in 'pVec' is located in any area 'size' of the 4 corners of 'image'
*/
bool checkBoundaries(std::vector<cv::Point> pVec, cv::Mat &image, cv::Size size)
{
    if(pVec.size() == 0)
        return false;
    
    // Creates 4 rects of size cv::Size size in 4 corners of the image
    std::vector<cv::Rect> rectVec;
    rectVec.push_back(cv::Rect(cv::Point(0,0),size));
    rectVec.push_back(cv::Rect(cv::Point(image.cols-size.width,0),size));
    rectVec.push_back(cv::Rect(cv::Point(0,image.rows-size.height),size));
    rectVec.push_back(cv::Rect(cv::Point(image.cols-size.width,image.rows-size.height),size));
    
    //Draw detected circles in image
    std::vector<cv::Point>::const_iterator itp = pVec.begin();
    while(itp != pVec.end())
    {
        std::vector<cv::Rect>::const_iterator itr = rectVec.begin();
        while(itr != rectVec.end())
        {
            if(itr[0].contains(itp[0]))
                return true;
            itr++;
        }    
        itp++;
    }
    
    return false;
}

/**
    Detects traffic sign numbers
    - Detect contours from cropped traffic sign image
    - Filters contours by area (size) and by detecting if they touch a corner of the image
    - Dilates the image resulting from the previous operation and aplies mask to original image.
*/
void contourFilter(std::vector<cv::Mat> &maskedVector)
{
    if(maskedVector.size() == 0)
        return ;

    cv::namedWindow("inter",cv::WINDOW_NORMAL);
   cv::imshow("inter", maskedVector[0]);
    
    // Finds contourts from image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(maskedVector[0], contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0,0));
    
    // Draw filtered contours in 'contourImg', filtering by area and by checking if they touch a corner of the image
    cv::Mat contourImg = cv::Mat::zeros(cv::Size(maskedVector[0].cols, maskedVector[0].rows), CV_8UC1);
    for(int i = 0; i < (int)contours.size(); i++)
    {
        std::vector<cv::Point> contour = contours[i];
        double area = cv::contourArea(contour);
        if((area > minContourThresh)
            &&(area < maxContourThresh)
            &&!checkBoundaries(contour, maskedVector[0],
            cv::Size(maskedVector[0].cols*0.1,maskedVector[0].rows*0.1)))
            {
                cv::drawContours(contourImg, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0, cv::Point());
            }
    }
          
    // Dilates, to create mask for number detection
    cv::Mat dilate_element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                     cv::Size(dilateSize, dilateSize),
                     cv::Point(-1,-1));
    cv::dilate(contourImg, contourImg, dilate_element);
    contourImg = contourImg > 1;
    
    // Masks
    cv::Mat filtImg = cv::Mat::zeros(cv::Size(contourImg.cols, contourImg.rows), CV_8UC1);
    maskedVector[0].copyTo(filtImg,contourImg);
    filtImg = filtImg > 1;



/*
    // TODO? Find contour from filtered image and check intersection with original contours
    std::vector<std::vector<cv::Point>> filtContours;
    std::vector<cv::Vec4i> filtHierarchy;
    cv::findContours(filtImg, filtContours, filtHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0,0));

    // Check if filtered image contours intersect with original image contours
    std::vector<std::vector<cv::Point>> intersect;
    std::vector<std::vector<cv::Point>>::const_iterator itp = filtContours.begin();
    std::vector<std::vector<cv::Point>>::const_iterator itr = contours.begin();
    
    while(itp != filtContours.end())
    {
        while(itr != contours.end())
        {
            if(itr[0].contains(*itp))
            {
               intersect.push_back(*itr);
               break;
            }
            itr++;
        }
        itp++;
    }
    
    // Draw resultant contours
    cv::Mat interImg = cv::Mat::zeros(cv::Size(filtImg.cols, filtImg.rows), CV_8UC1);
    for(int i = 0; i < (int)intersect.size(); i++)
    {
        cv::drawContours(interImg, intersect, i, cv::Scalar(255,255,255), -1, 8, 0, 0, cv::Point());
    }
     

*/



    
    if(print)
    {
/*
        cv::namedWindow("inter",cv::WINDOW_NORMAL);
        cv::imshow("inter", interImg);
*/

        cv::imshow("Contour", contourImg);
        cv::imshow("OCR_Input", filtImg);
/*    
        cv::namedWindow("Skel",cv::WINDOW_NORMAL);
        cv::Mat skel = getSkel(maskedVector[0]);
        cv::imshow("Skel",skel);
        std::cout << "Skel masked : " << ocr(skel) << "\n";

        cv::Mat opn;
        cv::morphologyEx(filtImg,opn,cv::MORPH_GRADIENT, dilate_element);            
        cv::namedWindow("opn",cv::WINDOW_NORMAL);
        cv::imshow("opn",opn);
        std::cout << "Opn : " << ocr(opn) << "\n";        */
    }
    
    if(save)
    {
        cv::imwrite(file+std::string("_contour_")+"."+ext,contourImg);
        cv::imwrite(file+std::string("_filtered_")+"."+ext,filtImg);
    }   
    std::cout << "ROI : " << ocr(maskedVector[0]);
    std::cout << "Contour : " << ocr(contourImg);
    std::cout << "OCR Input : " << ocr(filtImg);
               
}

    
void process(cv::Mat &image)
{   
    if(print)
    {
        // hough() trackbar params
         dp = 1;  dpTbar = 1;
         minDist = 0;  minDistTbar = 100/8;
         param1 = 100;
         param2 = 50;
         minRadius = 10;
         maxRadius = 50;

         dpTbarPrev = 0;
         minDistTbarPrev = 0;
         param1Prev = 0;
         param2Prev = 0;
         minRadiusPrev = 0;
         maxRadiusPrev = 0;

        // contourFilter() trackbar params
         minContourThresh = 15;
         maxContourThresh = 900;
         dilateSize = 3;
         minContourThreshPrev = 15;
         maxContourThreshPrev = 900;
         dilateSizePrev = 3;

        cv::namedWindow("Hough",cv::WINDOW_NORMAL);
        cv::createTrackbar("DP", "Hough", &dpTbar, 100, NULL);
        cv::createTrackbar("MinDist", "Hough", &minDistTbar, 100, NULL);
        cv::createTrackbar("CannyThresh", "Hough", &param1, 255, NULL);
        cv::createTrackbar("Accum", "Hough", &param2, 1000, NULL);
        cv::createTrackbar("MinRadius", "Hough", &minRadius, image.rows, NULL);
        cv::createTrackbar("MaxRadius", "Hough", &maxRadius, image.rows, NULL);
        
        cv::namedWindow("Circles",cv::WINDOW_NORMAL);
        
        cv::namedWindow("ROI",cv::WINDOW_NORMAL);
        cv::namedWindow("OCR_Input",cv::WINDOW_NORMAL);
        cv::namedWindow("Contour",cv::WINDOW_NORMAL);
        cv::createTrackbar("MinContour", "Contour", &minContourThresh, 1000, NULL);
        cv::createTrackbar("MaxContour", "Contour", &maxContourThresh, 1000, NULL);
        cv::createTrackbar("dilateSize", "Contour", &dilateSize, 255, NULL);         
    }

    char c = 0;
    std::vector<cv::Vec3f> circles = {};
    std::vector<cv::Mat> maskVector = {};

    while(c != 27)
    {
        if(houghTrackbar()||contourTrackbar())
        {
            circles = hough(image);
            maskVector = crop(image,circles);
            contourFilter(maskVector);
        }    
        if((!print && !singleImage) || (!singleImage && (circles.size() == 0)))
            break;

        c = cv::waitKey(50);
    }
}


int main(int argc, char **argv)
{
    print = false;
    save = false;
    singleImage = false;
    video = false;
    
    if(argc > 5)
    {
        std::cerr << "Use: $./sign_contour o $./sign_contour -i image; use -p to display images, and -s to save them.\n";
        return -1;
    } 
    
    cv::Mat image;
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "-p")
        {
            print = true;
        }
        if (std::string(argv[i]) == "-s")
        {
            save = true;
        }
        else if (std::string(argv[i]) == "-i")
        {
            singleImage = true;
            if(i+1 <= argc)
            {
                parseFilename(argv[i+1]);
            }    
            else
                return -1;
        }
        else if (std::string(argv[i]) == "-v")
        {
            video = true;
            if(i+1 <= argc)
                file = std::string(argv[i+1]);
            else
                return -1;
        }
    }

    if(singleImage)
    {
        image = cv::imread(file+"."+ext,CV_LOAD_IMAGE_UNCHANGED);
        if(!image.data) // Check for invalid input
        {
         std::cerr << "Could not open or find the image " << file << "\n";
         return -1;
        }
        process(image);
    }            
    else if(video)
    {
        cv::VideoCapture cap(file);
        if(!cap.isOpened())
        {
            std::cerr << "Could not open or find the video " << file << "\n";
            return -1;
        }

        double fps = cap.get(CV_CAP_PROP_FPS);
        std::cout << "Frame per seconds : " << fps << "\n";

        int i = 0;
        while(1)
        {
            cv::Mat frame;
            if (!cap.read(frame)) 
            {
                std::cerr << "Cannot read frame from video file.\n";
                break;
            }
            
            if(i%1 == 0)
            {
                frameNumber = i;
                process(frame);
            }

            if(cv::waitKey(1) == 27)
                break;
            i++;
        }
    }
    else
    {
        for(int i=1;i<35;i++) // Circles through test images
        {
            std::string filename = "image-";
            std::ostringstream stream;
            stream << std::setw(3) << std::setfill('0') << i;
            filename += stream.str();
            filename += ".jpg";
            parseFilename((char *)filename.c_str());
            image = cv::imread(file+"."+ext,CV_LOAD_IMAGE_UNCHANGED);
            if(!image.data)
            {
                std::cerr << "Could not open or find the image " << file << "\n";
                continue;
            }
            process(image);
        }
    }
    
    cv::destroyAllWindows();   
    return 0;
}



cv::Mat getSkel(cv::Mat &img)
{
    cv::threshold(img, img, 127, 255, cv::THRESH_BINARY); 
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;
 
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
 
    bool done;  
    do
    {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);
 
        done = (cv::countNonZero(img) == 0);
    } while (!done);
    return skel;
}
