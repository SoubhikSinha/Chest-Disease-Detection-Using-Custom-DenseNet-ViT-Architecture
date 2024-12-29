CSE 573 - Computer Vision and Image Processing
==============================================
-by Dr. Nalini Ratha

FINAL PROJECT (CAPSTONE)
Team🔻
SOUBHIK SINHA (soubhik : 50545730)
EKLAVYA (eklavya : 50545715)

INSTRUCTIONS FOR RUNNING THIS PROJECT ON YOUR SYSTEM 🔽
----------------------------------------------------
NOTE 🌟 : Due to size constraints on UBLearns (yeah, it only allows 2GB file size to upload - at a time),
We had to compress the Dataset Images as well. Thus, you need to extract "Datasets" ZIP file first and
include the extracted "Datasets/Datasets" ZIP file (folder) to "Code" Directory (Please keep this in mind that
when you try to extract - First you will witness an extracted "Datasets" folder. Once you go inside - you will find
another "Datasets" folder - copy this and paste it in the "Code" folder - such that it doesn't look like "Datasets/Datasets" !!!

1. Go to the "Code" Directory.

2. Open Git Bash (or CMD/Terminal if on Mac) in the "Code" directory.

3. Run the command: ./create.sh
This will create the CVIP_ENV Anaconda environment.
All dependencies from requirements.txt will be installed automatically.
If you can’t run .sh files in CMD/Terminal, use Git Bash or ensure your terminal supports it.

4. Run the command: ./run.sh
This will activate the CVIP_ENV environment and launch the Gradio app.
The IP address for the Gradio app will be printed in the terminal.

5. Copy the IP address (e.g., http://<your-ip>:7860) and paste it in your web browser’s address bar.

6. You can upload sample images (located in folder : "CVIP_PROJ_Deploy-Test Images") or any random image (Chest X-Ray, Lung CT-Scan) to test the application.

7. Once done, stop the server by pressing Ctrl+C in Git Bash/CMD/Terminal.