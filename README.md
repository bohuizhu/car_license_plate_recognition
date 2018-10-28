# car_license_plate_recognition project
bohui Zhu 
<p>This project includes basic algorithms for identifying car license plates characters for Vic Road company, the basic content includes car license plate generation, car license plate detection and extraction, license plate characters segmentation, car license plate characters recognition.</p>
<h1> Configuration </h1>
<p> python 3.5 tensorflow 3.5 </p>
<h1> car_license_plate_generation algorithm </h1>
<p> the algorithm tries to build some jpg images by randomy choosing the permuation of car license plate characters. to run the algorithm, simply type python car_license_plate_generation on the shell, this algorithms uses reference from https://blog.csdn.net/Gavinmiaoc/article/details/79412174 </p>
<h1> car_license_plate_detection algorithm</h1>
<p> this algorithms tries to select a frame from an image that possible contains the number plate and then extraction of the region of the number plate, to run the simply type python car_license_plate_detection on the shell.</p>
<h1> car license plate segmentation algorithm</h1>
<p> this algorithm tries to detect and segment single characters inside the car license plate, and generate corresponding characters images, to run the algorithm simply type python car_plate_segmentation.py. The algorithms use reference from https://blog.csdn.net/m0_38024433/article/details/78650024?utm_source=blogxgwz1 </p>
<h1> car license charater recognition algorithms -- CNN model </h1>
<p> This algorithm tries to build a convolutional neural network for identifying car license plate characters, it contains two parts:training and prediction. To load the algorithm, first need to run python car_license_plate_characers_recognition.py train to train the CNN model, and run python car_license_plate_characers_recognition.py predict to predict the tested image inside the algorithm.</p>

<h2>input declarsion </h2>
<p> all the algorithm inputs are fed inside the algorithm, if need to change different input image to test, need to change the input filepath inside each algorithm.</p>

