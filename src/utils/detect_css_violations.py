from .metrics import associate_score

class STrack(object):
    def __init__(self, tlwh, score, track_id):
        self.tlwh = tlwh
        self.score = score
        self.track_id = track_id
        self.missing = []  # List to store missing equipment IDs

def detect_css_violations(online_targets, obj_detections, iou_threshold=0.5):
    REQUIRED_EQUIPMENT = {0: "Hardhat", 2: "Safety Vest", 4: "Gloves"}
    res = []
    for t in online_targets:
        # Convert tlwh to xyxy
        new_track = STrack(t.tlwh, t.score, t.track_id)
        x1 = t.tlwh[0]
        y1 = t.tlwh[1]
        x2 = x1 + t.tlwh[2] 
        y2 = y1 + t.tlwh[3] 
        person_bbox = [x1, y1, x2, y2]
        print("===Checking person", t.track_id, "with bbox:", person_bbox)
        # Track what equipment is found
        found_equipment = set()
        
        # Check each detected object
        for obj in obj_detections:
            obj_bbox = obj[:4]  # x1,y1,x2,y2
            obj_class = obj[5]  # class_id
            # print("===Checking object:", obj_class, "with bbox:", obj_bbox)

            # Only check required safety equipment
            if obj_class not in REQUIRED_EQUIPMENT:
                continue    
            if associate_score(person_bbox, obj_bbox) > iou_threshold:
                found_equipment.add(obj_class)
            # print("associate_score results:", associate_score(person_bbox, obj_bbox))
        
        # add missing equipment to the person (we track person)
        new_track.missing = [eq_id for eq_id in REQUIRED_EQUIPMENT.keys() 
                    if eq_id not in found_equipment]
        res.append(new_track)  
    return res