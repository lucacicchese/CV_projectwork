# CV_projectwork

## Contents

1. [Completed tasks](#completed-tasks)
2. [Environment](#environment)
3. [Implementation](#implementation)

## Completed tasks

|  Feature   | DONE  | WIP |
|-----|---|---|
| My indoor dataset |  | 🔁 |
| My outdoor dataset |  | 🔁 |
| Extract poses colmap | ✅ |  |
| Extract 3d points colmap | ✅ |  |
| Extract poses mast3r | ✅ |  |
| Extract 3d points mast3r | ✅ |  |
| Extract poses vggt | ✅ |  |
| Extract 3d points vggt | ✅ |  |

## Environment

To work on use the code it's necessary to follow these steps:

- create a root project folder
- clone this repository in the root folder
- clone mast3r repository in the root folder <https://github.com/naver/mast3r>
- clone VGGT repository in the root folder <https://github.com/facebookresearch/vggt>
- install COLMAP <https://colmap.github.io/index.html>

At the end of these steps you should have the following file structure:

```
root
|
|-- CV_projectwork
|-- mast3r
|-- vggt
```

The other requirements to run the code can be installed by running the requirements.txt file included in the CV_projectwork folder:
`pip install -r requirements.txt`

To run the full pipeline just run the main file contained in the CV_projectwork folder:
`python main.py`

## Datasets

### Indoor dataset

### Outdoor dataset

## Implementation

### main.py

### extract.py

### metrics.py

### splatting.py
