#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>

#include <opencv2/ml.hpp>

#include <opencv2/imgproc.hpp>



#include <iostream>

#include <fstream>

#include <vector>



using namespace cv;

using namespace cv::ml;

using namespace std;



const int IMAGE_SIZE = 28;



//load images

vector<vector<unsigned char>> LoadImages(const string& filePath) {

    ifstream file(filePath, ios::binary);



    char header[4], numImagesBytes[4], numRowsBytes[4], numColsBytes[4];

    file.read(header, 4), file.read(numImagesBytes, 4), file.read(numRowsBytes, 4), file.read(numColsBytes, 4);



    int numImages = (static_cast<unsigned char>(numImagesBytes[3]) << 0) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[0]) << 24);

    int numRows = (static_cast<unsigned>(numRowsBytes[3]) << 0) | (static_cast<unsigned>(numRowsBytes[2]) << 8) | (static_cast<unsigned>(numRowsBytes[1]) << 16) | (static_cast<unsigned>(numRowsBytes[0]) << 24);

    int numCols = (static_cast<unsigned>(numColsBytes[3]) << 0) | (static_cast<unsigned>(numColsBytes[2]) << 8) | (static_cast<unsigned>(numColsBytes[1]) << 16) | (static_cast<unsigned>(numColsBytes[0]) << 24);



    vector<vector< unsigned char >>  result;



    for (int i = 0; i < numImages; i++) {

        vector<unsigned char> image(numRows * numCols);

        file.read(reinterpret_cast<char*>(image.data()), numRows * numCols);

        result.push_back(image);

    }

    file.close();

    return result;

}



//load labels

vector<vector<unsigned char>> LoadLabels(const string& filePath) {

    ifstream file(filePath, ios::binary);



    char header[4], numLabelsBytes[4];

    file.read(header, 4), file.read(numLabelsBytes, 4);



    int numLabels = (static_cast<unsigned char>(numLabelsBytes[3]) << 0) | (static_cast<unsigned char>(numLabelsBytes[2]) << 8) | (static_cast<unsigned char>(numLabelsBytes[1]) << 16) | (static_cast<unsigned char>(numLabelsBytes[0]) << 24);



    vector<vector<unsigned char>> result;



    for (int i = 0; i < numLabels; i++) {

        vector<unsigned char> label(1);

        file.read(reinterpret_cast<char*>(label.data()), 1);

        result.push_back(label);

    }

    file.close();

    return result;

}

Mat PreprocessImage(const Mat& inputImage) {
    Mat preprocessedImage;

//    for smoothing
//    GaussianBlur(inputImage, preprocessedImage, Size(5, 5), 0);
//
//    convert the image to grayscale
//    cvtColor(preprocessedImage, preprocessedImage, COLOR_BGR2GRAY);
//
//    enhance edges
//    threshold(preprocessedImage, preprocessedImage, 128, 255, THRESH_BINARY);
//
//    other preprocessing steps can be added here

    return preprocessedImage;
}

//train KNN model

Ptr<KNearest> TrainKNN(const vector<vector<unsigned char>>& images, const vector<vector<unsigned char>>& labels) {

    Mat trainingData(images.size(), IMAGE_SIZE * IMAGE_SIZE, CV_32F), labelsMat(labels.size(), 1, CV_32S);



    for (size_t i = 0; i < images.size(); ++i) {

        Mat imageMat(IMAGE_SIZE, IMAGE_SIZE, CV_8U, const_cast<uchar*>(images[i].data())), imageMat32F;

        imageMat.convertTo(imageMat32F, CV_32F);

        imageMat32F.reshape(1, 1).copyTo(trainingData.row(static_cast<int>(i)));

        labelsMat.at<int>(static_cast<int>(i), 0) = labels[i][0];

    }



    Ptr<KNearest> knn = KNearest::create();

    knn->train(trainingData, ROW_SAMPLE, labelsMat);



    return knn;

}



//recognize digit using the trained KNN model

int RecognizeDigit(const Mat& testImage, const Ptr<KNearest>& knn) {

    Mat testImage32F;

    testImage.convertTo(testImage32F, CV_32F);

    Mat testImageReshaped = testImage32F.reshape(1, 1), results, neighborResponses, dists;

    knn->findNearest(testImageReshaped, 1, results, neighborResponses, dists);



    return static_cast<int>(results.at<float>(0, 0));

}



//evaluate and display accuracy

void EvaluateAccuracy(const vector<vector<unsigned char>>& testImages, const vector<vector<unsigned char>>& testLabels, const Ptr<KNearest>& knn) {

    int correctPredictions{};



    for (size_t i = 0; i < testImages.size(); i++) {

        Mat testImageMat(IMAGE_SIZE, IMAGE_SIZE, CV_8U, const_cast<uchar*>(testImages[i].data()));

        resize(testImageMat, testImageMat, Size(IMAGE_SIZE, IMAGE_SIZE));

        // Preprocess the test image
        //Mat preprocessedImage = PreprocessImage(testImageMat);

        int recognizedDigit = RecognizeDigit(testImageMat, knn);

        int actualDigit = static_cast<int>(testLabels[i][0]);



        cout << "Digit: " << actualDigit << "               Using KNN: " << recognizedDigit << endl;



        if (recognizedDigit == actualDigit) {

            correctPredictions++;

        }

        imshow("Test Image", testImageMat);

    }



    //calculate and display accuracy

    double accuracy = (static_cast<double>(correctPredictions) / testImages.size() * 100);

    cout << "Accuracy is indeed " << accuracy << "%" << endl;

    //cout << "Accuracy is indeed " << "100%" << endl;

}



int main() {

    vector<vector<unsigned char>> trainImages = LoadImages("/Users/filberthamijoyo/Downloads/train-images.idx3-ubyte"), trainLabels = LoadLabels("/Users/filberthamijoyo/Documents/train-labels.idx1-ubyte"),  testImages = LoadImages("/Users/filberthamijoyo/Documents/t10k-images.idx3-ubyte"), testLabels = LoadLabels("/Users/filberthamijoyo/Documents/t10k-labels.idx1-ubyte");

    

    //training

    Ptr<KNearest> knn = TrainKNN(trainImages, trainLabels);

    

    //evaluate and display accuracy

    EvaluateAccuracy(testImages, testLabels, knn);

    return 0;

}

