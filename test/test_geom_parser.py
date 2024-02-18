from geom_parser import *

geom = GEOM(r"E:\BigFiles\GENAC\GENACX50kV\optimizing\GenacX50kV_tmplt_20240210_051249_02.m2d")
def test_export_geometry():
    export_geometry(geom)
    # plt.show()