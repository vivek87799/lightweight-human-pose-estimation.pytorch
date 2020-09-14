"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

import numpy as np

from openvino.inference_engine import IECore


class InferenceEngine:
    def __init__(self, net_model_xml_path, device, stride):
        self.device = device
        self.stride = stride

        self.ie = IECore()
        self.net = self.ie.read_network(net_model_xml_path, os.path.splitext(net_model_xml_path)[0] + '.bin')
        required_input_key = {'data'}
        assert required_input_key == set(self.net.input_info), \
            'Demo supports only topologies with the following input key: {}'.format(', '.join(required_input_key))
        # required_output_keys = {'features', 'heatmaps', 'pafs'}
        required_output_keys = {'stage_1_output_0_pafs', 'stage_1_output_1_heatmaps'}
        print("output keys..", self.net.outputs.keys())
        assert required_output_keys.issubset(self.net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        ## Get the available devices for multi config inference 
        
        ## all_device = ""
        """
        for _device in self.ie.available_devices:
            if _device == "CPU":
                continue
            if all_device == "":
                all_device = _device
            else:
                all_device = all_device+","+_device
        all_device = "MULTI:"+all_device
        """
        # print("AVAILABLE DEVICES ARE ", all_device)
        # self.exec_net = self.ie.load_network(network=self.net, device_name=device)
        self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=device)
        # self.ie.set_config({"MULTI_DEVICE_PRIORITIES": "MYRIAD.3.1.4-ma2480, MYRIAD.3.2-ma2480"}, "MULTI")
        # self.exec_net = self.ie.load_network(self.net, all_device, {})
        # self.exec_net = self.ie.load_network(self.net, "MULTI:MYRIAD.3.1-ma2480", {})

    def infer(self, img):
        """
        img = img[0:img.shape[0] - (img.shape[0] % self.stride),
                  0:img.shape[1] - (img.shape[1] % self.stride)]
        """
        input_layer = next(iter(self.net.input_info))
        n, c, h, w = self.net.input_info[input_layer].input_data.shape
        
        inference_result = self.exec_net.infer(inputs={'data': img})
        return inference_result