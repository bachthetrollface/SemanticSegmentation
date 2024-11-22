# Semantic Segmentation Exercise
- To clone this repository: git clone https://github.com/bachthetrollface/SemanticSegmentation.git
- To run the inference:
    + Download the model state dict: https://drive.google.com/file/d/1jxesnxJqwZseOPCaqPvfvGZX2VzN2Rh6/view?usp=sharing and place the file right in the SemanticSegmentation directory, the same place you can find the infer.py file.
    + Move to the directory: cd SemanticSegmentation
    + Check for requirements: you can manually check the file requirements.txt, or run the following: pip install -r requirements.txt
    + Run inference: python3 infer.py --image_path image.jpeg