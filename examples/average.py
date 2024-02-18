from grd_parser import GRD
import numpy

def range_time_average(ranges):
    rg_avg = numpy.average(numpy.array([rg['data'][1] for rg in ranges]), axis=0)
    return numpy.array([ranges[0]['data'][0],rg_avg])

