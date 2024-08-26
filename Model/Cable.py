class Cable:
    def __init__(self, Info, name, Wires, Phase, phase_num, ground, Measurement):
        """
        初始化电缆对象

        参数:
        Info (CableInfo): 电缆自描述信息对象
        Wires (Wires): 电缆线段对象集合
        Ground (Ground): 电缆地线对象集合
        Measurement (Measurement): 电缆测量对象集合
        """
        self.name = name
        self.wires = Wires
        self.info = Info
        self.Phase = Phase
        self.phase_num = phase_num
        self.ground = ground
        self.wires_name = []
        self.nodes_name = []
        # 以下是参数矩阵，是OHL建模最终输出的参数
        # 邻接矩阵 (pandas.Dataframe,self.wires_name*self.nodes_name)
        self.incidence_matrix = None
        # 电阻矩阵 (pandas.Dataframe,self.wires_name*self.wires_name)
        self.resistance_matrix = None
        # 电感矩阵 (pandas.Dataframe,self.wires_name*self.wires_name)
        self.inductance_matrix = None
        # 电容矩阵 (pandas.Dataframe,self.nodes_name*self.nodes_name)
        self.capacitance_matrix = None
        # 电导矩阵 (pandas.Dataframe,self.nodes_name*self.nodes_name)
        self.conductance_matrix = None
        # 阻抗矩阵 (pandas.Dataframe,self.wires_name*(self.wires_name*frequency_num))
        self.impedance_martix = None

    def get_brans_nodes_list(self, segment_num):
        """
        【函数功能】 获取切分后支路列表与节点列表
        【入参】
        segment_num(int):一条线切分后的线段数
        segment_length(float):线长

        【出参】
        brans_name(list,wires_num*segment_num):支路名称列表
        nodes_name(list,wires_num*segment_num+1):节点名称列表
        """
        brans_name = self.wires.get_all_wires()
        start_nodes_name = self.wires.get_all_start_nodes()
        end_nodes_name = self.wires.get_all_end_nodes()
        if segment_num == 1:
            self.wires_name = brans_name
            self.nodes_name = start_nodes_name
            self.nodes_name.extend(end_nodes_name)
        else:
            for i in range(segment_num):
                for j in range(len(brans_name)):
                    self.wires_name.append(f"{brans_name[j]}_Splited_{i+1}")

                self.nodes_name.extend(start_nodes_name)
                for j in range(len(start_nodes_name)):
                    start_nodes_name[j] = f"{brans_name[j]}_MiddleNode_{i + 1}"

            # 最后一个分段，则将终止节点加入列表
            self.nodes_name.extend(end_nodes_name)