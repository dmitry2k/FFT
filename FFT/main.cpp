#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;


void shiftDFT(Mat& fImage)
{
	Mat tmp, q0, q1, q2, q3;

	// first crop the image, if it has an odd number of rows or columns

	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

	int cx = fImage.cols / 2;
	int cy = fImage.rows / 2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center

	q0 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q2 = fImage(Rect(0, cy, cx, cy));
	q3 = fImage(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

/******************************************************************************/
// return a floating point spectrum magnitude image scaled for user viewing
// complexImg- input dft (2 channel floating point, Real + Imaginary fourier image)
// rearrange - perform rearrangement of DFT quadrants if true

// return value - pointer to output spectrum magnitude image scaled for user viewing

Mat create_spectrum_magnitude_display(Mat& complexImg, bool rearrange)
{
	Mat planes[2];

	// compute magnitude spectrum (N.B. for display)
	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))

	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);

	Mat mag = (planes[0]).clone();
	mag += Scalar::all(1);
	log(mag, mag);

	if (rearrange)
	{
		// re-arrange the quaderants
		shiftDFT(mag);
	}

	normalize(mag, mag, 0, 1, CV_MINMAX);

	return mag;

}


void create_ideal_lowpass_filter(Mat &dft_Filter, int D)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

	Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
	double radius;

	for (int i = 0; i < dft_Filter.rows; i++)
	{
		for (int j = 0; j < dft_Filter.cols; j++)
		{
			radius = (double)sqrt(pow((i - centre.x), 2.0) + pow((double)(j - centre.y), 2.0));
			tmp.at<float>(i, j) = radius > D ? 0.0f : 1.0f;
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dft_Filter);
}

void create_ideal_highpass_filter(Mat &dft_Filter, int D)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

	Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
	double radius;

	for (int i = 0; i < dft_Filter.rows; i++)
	{
		for (int j = 0; j < dft_Filter.cols; j++)
		{
			radius = (double)sqrt(pow((i - centre.x), 2.0) + pow((double)(j - centre.y), 2.0));
			tmp.at<float>(i, j) = radius > D ? 1.0f : 0.0f;
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dft_Filter);
}


int main(int argc, const char ** argv)
{
	Mat img = imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty())
	{
		printf("Cannot read image file: %s\n", "lena.jpg");
		return -1;
	}

	imshow("start image", img);

	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat complexImg2 = complexImg.clone();

	// create magnitude spectrum for display
	Mat mag = create_spectrum_magnitude_display(complexImg, true);


	//!!!!!!!!!!!!!!!!!!!
	//low pass filter   !
	//!!!!!!!!!!!!!!!!!!!

	Mat filter = complexImg.clone();
	create_ideal_lowpass_filter(filter, 100.0f);

	// apply filter
	shiftDFT(complexImg);
	mulSpectrums(complexImg, filter, complexImg, 0);
	shiftDFT(complexImg);

	// create magnitude spectrum for display
	Mat mag2 = create_spectrum_magnitude_display(complexImg, true);

	// do inverse DFT on filtered image
	idft(complexImg, complexImg);

	Mat imgOutput1, filterOutput1;

	// split into planes and extract plane 0 as output image
	split(complexImg, planes);
	normalize(planes[0], imgOutput1, 0, 1, CV_MINMAX);

	// do the same with the filter image
	split(filter, planes);
	normalize(planes[0], filterOutput1, 0, 1, CV_MINMAX);



	//!!!!!!!!!!!!!!!!!!!
	//high pass filter  !
	//!!!!!!!!!!!!!!!!!!!

	Mat filter2 = complexImg2.clone();
	create_ideal_highpass_filter(filter2, 10.0f);

	// apply filter2
	shiftDFT(complexImg2);
	mulSpectrums(complexImg2, filter2, complexImg2, 0);
	shiftDFT(complexImg2);

	// create magnitude spectrum for display
	Mat mag3 = create_spectrum_magnitude_display(complexImg2, true);

	// do inverse DFT on filtered image
	idft(complexImg2, complexImg2);

	Mat imgOutput2, filterOutput2;

	// split into planes and extract plane 0 as output image
	split(complexImg2, planes);
	normalize(planes[0], imgOutput2, 0, 1, CV_MINMAX);

	// do the same with the filter image
	split(filter2, planes);
	normalize(planes[0], filterOutput2, 0, 1, CV_MINMAX);

	// display images in window
	imshow("spectr of image", mag);
	imshow("spectr of image filtered by ideal low pass filter", mag2);
	imshow("image filtered by ideal low pass filter", imgOutput1);
	imshow("spectr of ideal low pass filter", filterOutput1);
	imshow("spectr of image filtered by ideal high pass filter", mag3);
	imshow("image filtered by ideal high pass filter", imgOutput2);
	imshow("spectr of ideal high pass filter", filterOutput2);

	waitKey();
	return 0;
}
