import json
from collections import namedtuple
Node = namedtuple('Node', ['centroid', 'bbox', 'contour', 'type', 'type_prob'])


class NucleiReader:

    def in_area(self, point, start_point, end_point):
        if start_point[0] < point[0] < end_point[0]:
            if start_point[1] < point[1] < end_point[1]:
                return True
        else:
            return False

    def clip(self, point, start_point, end_point):
        x, y = point
        x = min(max(start_point[0], x), end_point[0])
        y = min(max(start_point[1], y), end_point[1])
        return x, y

    def shift(self, point, start_point):
        return [point[0] - start_point[0], point[1] - start_point[1]]

    def crop(self, nodes, start_point, end_point):
        """
            start_point: (x, y)
            end_point: (x1, y1)
        """
        res = []
        for node in nodes:
            cen = node.centroid
            if not self.in_area(cen, start_point, end_point):
                continue

            cen = self.shift(cen, start_point)
            bbox = node.bbox
            b1 = self.clip(bbox[:2], start_point, end_point)
            b1 = self.shift(b1, start_point)
            b2 = self.clip(bbox[2:], start_point, end_point)
            b2 = self.shift(b2, start_point)
            bbox = [*b1, *b2]

            cnts = node.contour
            new_cnts = []
            for cnt in cnts:
                if self.in_area(cnt, start_point, end_point):
                    cnt = self.shift(cnt, start_point)
                    new_cnts.append(cnt)
            cnts = new_cnts


            res.append(Node(centroid=cen, bbox=bbox, contour=cnts, type=node.type, type_prob=node.type_prob))
        return res

    def read_json(self, json_path):
        res = []
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            #print(res.keys())
            #print(res['nuc']['2175'])
            #print(len(res['nuc'].keys()))
            for k, v in json_data['nuc'].items():
                #print(type(v))
                #print(v.keys())
                res.append(v)

        return res

    # def inverse_format(self, nodes, scale):
    #     res = []
    #     for node in nodes:
    #             cen = node['centroid'] # x, y
    #             cen = [int(c / scale)  for c in cen]
    #             cen = cen[::-1]

    #             bbox = node['bbox']
    #             # bbox : [min_y, min_x, max_y, max_x]
    #             #bbox = [b // 2 for b in sum(bbox, [])]
    #             bbox = [b for b in sum(bbox, [])] # y, x
    #             bbox = [int(b / scale) for b in bbox]

    #             #contour[:, 0] += bbox[0]
    #             #contour[:, 1] += bbox[1]
    #             contour += bbox[:2]
    #             res.append(Node(centroid=cen, bbox=bbox, contour=contour))

    def _formatnode(self, nodes):
        res = []
        for node in nodes:
            cen = node['centroid'] # x, y

            bbox = node['bbox']
            bbox = [b for b in sum(bbox, [])] # y, x
            cnt = node['contour']
            node_type = node['type']
            type_prob = node['type_prob']

            res.append(Node(centroid=cen, bbox=bbox, contour=cnt, type=node_type, type_prob=type_prob))

        return res

    def _inverse_formatnode(self, nodes):
        res = []
        for node in nodes:
            #cen = nodes.centeroid
            res.append(node._asdict())
            bbox = res[-1]['bbox']
            bbox = [bbox[:2], bbox[2:]]
            res[-1]['bbox'] = bbox
            #res[-1]['type_prob'] = None
            #res[-1]['type'] = None

        return res

    def _inverse_formatxy(self, nodes, scale):
        res = []
        for node in nodes:
            cen = node.centroid
            cen = [int(c * scale)  for c in cen]

            bbox = node.bbox
            bbox = [int(b * scale) for b in bbox]
            bbox[:2] = bbox[:2][::-1] # convert to x, y
            bbox[2:] = bbox[2:][::-1]

            contour = [[int(x * scale), int(y * scale)] for [x, y] in node.contour]
            #contour = node['contour']
            res.append(Node(centroid=cen, bbox=bbox, contour=contour, type=node.type, type_prob=node.type_prob))

        return res

    def save_json(self, path, nodes, mag=None):
        assert nodes
        new_dict = {}
        inst_info_dict = {}
        #print('node length:', len(nodes), type(nodes))
        for inst_id, node in enumerate(nodes):
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                    "bbox": node['bbox'],
                    "centroid": node['centroid'],
                    "contour": node['contour'],
                    "type_prob": node['type_prob'],
                    "type": node['type'],
            }

        json_dict = {"mag": mag, "nuc": inst_info_dict}  # to sync the format protocol
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return new_dict

    def _formatxy(self, nodes, scale):

        # all elements to (x, y)
        res = []
        for node in nodes:
                cen = node.centroid
                cen = [int(c / scale)  for c in cen]

                bbox = node.bbox
                bbox = [int(b / scale) for b in bbox]
                bbox[:2] = bbox[:2][::-1] # convert to x, y
                bbox[2:] = bbox[2:][::-1]

                contour = [[int(x / scale), int(y / scale)] for [x, y] in node.contour]
                #contour = node['contour']
                res.append(Node(centroid=cen, bbox=bbox, contour=contour, type=node.type, type_prob=node.type_prob))

        return res

    def json2node(self, label, scale):
        label = self._formatnode(label)
        label = self._formatxy(label, scale)
        return label

    def node2json(self, label, scale):
        label = self._inverse_formatxy(label, scale)
        label = self._inverse_formatnode(label)
        return label
