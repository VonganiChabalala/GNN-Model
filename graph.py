import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
from geographiclib.geodesic import Geodesic
import metpy.calc as mpcalc
from bresenham import bresenham
# without this, wandb causes error.
os.environ["WANDB_START_METHOD"] = "thread"

city_fp = os.path.join(proj_dir, 'data/SA Monitoring Stations.xlsx')
#terrain_fp = os.path.join(proj_dir, 'data/terrain.npy')

#terrain_fp = np.load(terrain_fp)
class Graph():
    def __init__(self):
        self.dist_thres = 150
        self.nodes = self._gen_nodes()
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)
        self.edge_index,self.adj,self.edge_attr = self._gen_edge_index()
        #self.edge_attr = self._gen_edges()

        self.edge_num = self.edge_index.shape[1]

    def _gen_nodes(self):
        nodes = OrderedDict()
        data = pd.read_excel(city_fp)
        for i in range(len(data)):
            
            '''For each station, information about the location (tuple of longitude and latitude coordinates);
            altitude, and station name are extracted and stored in a list which will be used to extract the 
            node features'''
            
            idx, city, lon, lat, alt  = i,data.loc[i][0],data.loc[i][1],data.loc[i][2],data.loc[i][3]
            nodes.update({idx: {'Station': city, 'altitude': alt, 'location': (lon,lat)}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr


    def _gen_edge_index(self):
        distance_array = []
        angle_array = []
        total_peaks = []
        #ter_lon = terrain_fp[0]
        #ter_lat = terrain_fp[1]
        for i in range(len(self.nodes)):
            dist__ = []
            angs = []
            peaks_inbetween = []
            source_lon = self.nodes[i]['location'][0]
            source_lat = self.nodes[i]['location'][1]
            for j in range(len(self.nodes)):
                peaks = []
                receiver_lon = self.nodes[j]['location'][0]
                receiver_lat = self.nodes[j]['location'][1]
                geo_dict = Geodesic.WGS84.Inverse(source_lat,source_lon,receiver_lat,receiver_lon)
                distance = geo_dict['s12']/1000
                dist__.append(distance)
                bearing = 90 - geo_dict['azi1']
                if bearing < 0: bearing += 360
                bearing = int(bearing)
                angs.append(bearing)
                #for k in range(len(ter_lat)):
                    #geo1 = #Geodesic.WGS84.Inverse(ter_lat[k],ter_lon[k],receiver_lat,receiver_lon)
                    #dis1 = geo1['s12']/1000
                    #bear1 = 90 - geo1['azi1']
                    #if bear1 < 0: bear1 += 360
                    #bear1 =  int(bear1)
                    #geo2 = Geodesic.WGS84.Inverse(ter_lat[k],ter_lon[k],source_lat,source_lon)
                    #dis2 = geo2['s12']/1000
                    #bear2 = 90 - geo2['azi1']
                    #if bear2 < 0: bear2 += 360
                    #bear2 =  int(bear2)
            
                    #if dis1 < distance:
                        #if  bearing -5 < bear1 < bearing+5: 
                            #peaks.append(1)
                    
                    #elif dis2 < distance:
                        #if bearing -5 < bear2 < bearing+5:
                            #peaks.append(1)
                #peaks_inbetween.append(len(peaks))
            distance_array.append(dist__)
            angle_array.append(angs)
            #total_peaks.append(peaks_inbetween)


        
        distance_array = np.array(distance_array)
        angle_array = np.array(angle_array)
        distance_array = distance_array.reshape(len(self.nodes),len(self.nodes))
        angle_array = angle_array.reshape(len(self.nodes),len(self.nodes))
        #total_peaks = np.array(total_peaks)
        #total_peaks = total_peaks.reshape(len(self.nodes),len(self.nodes))

        dist_thresh = self.dist_thres#distance_array.mean() #change value to change how the graph looks
        adj1 = np.zeros((len(self.nodes),len(self.nodes)), dtype=np.uint8)
        #adj2 = np.zeros((len(self.nodes),len(self.nodes)), dtype=np.uint8)
        adj1[distance_array <= dist_thresh] = 1

        #adj2[total_peaks == 0] = 1

        #adj = adj1*adj2

        dist_arr = distance_array * adj1

        #angle_arr = angle_array*adj
        edge_index, dist_arr= dense_to_sparse(torch.tensor(dist_arr))
        edge_index, dist_arr = edge_index.numpy(), dist_arr.numpy()
        angles = []

        for i in range(edge_index.shape[1]):
            angles.append(angle_array[edge_index[0][i],edge_index[1][i]])
        angles = np.array(angles)
        angles = np.stack(angles)
        dist_arr = np.stack(dist_arr)
        attribs = np.stack([dist_arr,angles],axis=-1)

        edge_index = torch.tensor(edge_index)
        return edge_index,adj1,attribs

    '''def _gen_edges(self):
        distance_array = []
        angle_array = []
        for i in range(len(self.nodes)):
            #dist__ = []
            #angle__ = []
            source_lon = self.nodes[i]['location'][0]
            source_lat = self.nodes[i]['location'][1]
            for j in range(len(self.nodes)):
                receiver_lon = self.nodes[j]['location'][0]
                receiver_lat = self.nodes[j]['location'][1]
                geo_dict = Geodesic.WGS84.Inverse(source_lat,source_lon,receiver_lat,receiver_lon)
                distance_array.append(geo_dict['s12']/1000)
                bearing = 90 - geo_dict['azi1']
                if bearing < 0: bearing += 360
                angle_array.append(bearing)
        distance_array = np.stack(distance_array)
        angle_array = np.stack(angle_array)
        attr = np.stack([distance_array, angle_array], axis = -1)
        return torch.tensor(attr)'''

    


if __name__ == '__main__':
    graph = Graph()
