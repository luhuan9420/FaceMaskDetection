import os
from xml.etree import ElementTree
from tqdm import tqdm

class_name_to_id = {'with_mask': 0, 
					'mask_weared_incorrect': 1, 
					'without_mask': 2}

def xml_info(xml_file):
	root = ElementTree.parse(xml_file).getroot()

	info_dict = {}
	info_dict['bboxes'] = []

	for element in root:
		if element.tag == "filename":
			info_dict['filename'] = element.text

		elif element.tag == "size":
			image_size = []
			for attr in element:
				image_size.append(int(attr.text))
			info_dict['image_size'] = tuple(image_size)

		elif element.tag == "object":
			bbox = {}
			for attr in element:
				if attr.tag == "name":
					bbox['class'] = attr.text
				elif attr.tag == "bndbox":
					for dim in attr:
						bbox[dim.tag] = int(dim.text)
			info_dict['bboxes'].append(bbox)
	return info_dict

def convert_to_yolo(info_dict):
	print_buffer = []

	for b in info_dict['bboxes']:
		try:
			class_id = class_name_to_id[b['class']]
		except KeyError:
			print("Invalid Class")

		center_x = (b["xmin"] + b["xmax"]) / 2 
		center_y = (b["ymin"] + b["ymax"]) / 2
		width    = (b["xmax"] - b["xmin"])
		height   = (b["ymax"] - b["ymin"])

		image_w, image_h, _ = info_dict["image_size"]

		center_x /= image_w
		center_y /= image_h
		width    /= image_w
		height   /= image_h 

		print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, center_x, center_y, width, height))

	f_name = os.path.join("label_yolo", info_dict["filename"].replace('png', 'txt'))
	print("\n".join(print_buffer), file= open(f_name, "w"))


anno_path = '../data/annotations'
annotations = [os.path.join(anno_path, x) for x in os.listdir(anno_path) if x[-3:] == "xml"]
annotations.sort()

if not os.path.exists('label_yolo'):
	os.mkdir('label_yolo')

for f in tqdm(annotations):
	info_dict = xml_info(f)
	convert_to_yolo(info_dict)
