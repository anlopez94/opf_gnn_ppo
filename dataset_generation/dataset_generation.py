import pandapower.networks
import pandapower

net = pandapower.networks.case9()
print(net)
pandapower.to_excel(net, 'dataset/case9.xlsx')
solution = pandapower.runpp(net)
pandapower.to_excel(net, 'dataset/case9_pf.xlsx')

solution = pandapower.runopp(net)
pandapower.to_excel(net, 'dataset/case9_opf.xlsx')






#
# net = pandapower.converter.from_mpc('Data_Generation/PowerModels/test_5.mat')
#
# # print(net['bus'])
# # print(net['dcbus'])
# # print(net['load'])
# # print(net['gen'])
# print(net)
#
# solution = pandapower.runpp(net)
#
# print(net['sgen'])
# print(net['ext_grid'])