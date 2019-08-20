"""Simulation of the effect of surface errors on MID observations

This measures the effect of surface errors on the change in a dirty image induced by pointing errors:
    - The pointing errors can be random per integration, static, or global, or drawn from power spectra
    - The sky can be a point source at the half power point or a realistic sky constructed from S3-SEX catalog.
    - The observation is by MID over a range of hour angles
    - Processing can be divided into chunks of time (default 1800s)
    - Dask is used to distribute the processing over a number of workers.
    - Various plots are produced, The primary output is a csv file containing information about the statistics of
    the residual images.

"""
import csv
import os
import socket
import sys
import time

import seqfile

from data_models.parameters import arl_path

results_dir = arl_path('test_results')

import numpy

from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import Skycomponent, SkyModel

from processing_library.image.operations import create_empty_image_like
from wrappers.serial.visibility.base import create_blockvisibility

from wrappers.serial.image.operations import show_image, qa_image, export_image_to_fits
from wrappers.serial.simulation.configurations import create_configuration_from_MIDfile
from wrappers.serial.image.operations import import_image_from_fits
from wrappers.serial.imaging.primary_beams import create_pb
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent
from wrappers.serial.skycomponent.base import copy_skycomponent
from processing_library.util.coordinate_support import hadec_to_azel
from wrappers.serial.simulation.pointing import simulate_gaintable_from_pointingtable

from wrappers.serial.simulation.testing_support import simulate_pointingtable, simulate_pointingtable_from_timeseries
from wrappers.serial.imaging.primary_beams import create_vp, create_pb
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.serial.calibration.pointing import create_pointingtable_from_blockvisibility

from wrappers.arlexecute.visibility.base import copy_visibility
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility

from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_compsonly_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import weight_list_arlexecute_workflow
from workflows.shared.imaging.imaging_shared import sum_invert_results

from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client

import logging

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import pprint

pp = pprint.PrettyPrinter()


# Process a set of BlockVisibility's, creating pointing errors, converting to gainables, applying
# the gaintables to the FT of the skycomponents, and dirty images, one per BlockVisibility
def create_vis_list_with_errors(sub_bvis_list, sub_components, sub_model_list, vp_directory, use_radec=False):

    def find_vp(vis):
        ha = numpy.pi * numpy.average(vis.time) / 43200.0
        dec = vis.phasecentre.dec.rad
        latitude = vis.configuration.location.lat.rad
        az, el = hadec_to_azel(ha, dec, latitude)
        
        el_deg = el * 180.0 / numpy.pi
        
        if el_deg < 30.0:
            el_deg = 15.0
        elif el_deg < (90.0 + 45.0) / 2.0:
            el_deg = 45.0
        else:
            el_deg = 90.0
        
        # /Users/timcornwell/Code/sim-mid-pointing/beam_models/EMSS/with_elevation/B2_15_1360_imag.fits
        vpa = import_image_from_fits('%s/B2_%d_1360_real.fits' % (vp_directory, int(el_deg)))
        vpa_imag = import_image_from_fits('%s/B2_%d_1360_real.fits' % (vp_directory, int(el_deg)))
        vpa.data = vpa.data + 1j * vpa_imag.data
        return vpa
    
    def find_vp_nominal():
        el_nominal_deg = 45.0
        vpa_nominal = import_image_from_fits('%s/B2_%d_1360_real.fits' % (vp_directory, int(el_nominal_deg)))
        vpa_nominal_imag = import_image_from_fits('%s/B2_%d_1360_real.fits' % (vp_directory, int(el_nominal_deg)))
        vpa_nominal.data = vpa_nominal.data + 1j * vpa_nominal_imag.data
        return vpa_nominal

    error_pt_list = [arlexecute.execute(create_pointingtable_from_blockvisibility)(bvis) for bvis in sub_bvis_list]
    no_error_pt_list = [arlexecute.execute(create_pointingtable_from_blockvisibility)(bvis) for bvis in sub_bvis_list]

    vp_nominal_list = [arlexecute.execute(find_vp_nominal)() for bv in sub_bvis_list]
    vp_actual_list = [arlexecute.execute(find_vp)(bv) for bv in sub_bvis_list]

    # Create the gain tables, one per Visibility and per component
    no_error_gt_list = [arlexecute.execute(simulate_gaintable_from_pointingtable)
                        (bvis, sub_components, no_error_pt_list[ibv], vp_nominal_list[ibv], use_radec=use_radec)
                        for ibv, bvis in enumerate(sub_bvis_list)]
    error_gt_list = [arlexecute.execute(simulate_gaintable_from_pointingtable)
                     (bvis, sub_components, error_pt_list[ibv], vp_actual_list[ibv], use_radec=use_radec)
                     for ibv, bvis in enumerate(sub_bvis_list)]
    if show:
        tmp_gt_list = arlexecute.compute(error_gt_list, sync=True)
        plt.clf()
        for gt in tmp_gt_list:
            amp = numpy.abs(gt[0].gain[:, 0, 0, 0, 0])
            plt.plot(gt[0].time[amp > 0.0], 1.0 / amp[amp > 0.0], '.')
        plt.ylim([1e-1, 1.1])
        plt.title("%s: dish 0 amplitude gain" % basename)
        plt.xlabel('Time (s)')
        plt.savefig('gaintable.png')
        plt.show(block=False)

    # Each component in original components becomes a separate skymodel
    # Inner nest is over skymodels, outer is over bvis's
    error_sm_list = [[
        arlexecute.execute(SkyModel, nout=1)(components=[sub_components[i]], gaintable=error_gt_list[ibv][i])
        for i, _ in enumerate(sub_components)] for ibv, bv in enumerate(sub_bvis_list)]

    no_error_sm_list = [[
        arlexecute.execute(SkyModel, nout=1)(components=[sub_components[i]], gaintable=no_error_gt_list[ibv][i])
        for i, _ in enumerate(sub_components)] for ibv, bv in enumerate(sub_bvis_list)]

    # Predict each visibility for each skymodel. We keep all the visibilities separate
    # and add up dirty images at the end of processing. We calibrate which applies the voltage pattern
    no_error_bvis_list = [arlexecute.execute(copy_visibility, nout=1)(bvis, zero=True) for bvis in sub_bvis_list]
    no_error_bvis_list = [
        predict_skymodel_list_compsonly_arlexecute_workflow(no_error_bvis_list[ibv],
                                                            no_error_sm_list[ibv],
                                                            context='2d', docal=True)
        for ibv, bvis in enumerate(no_error_bvis_list)]
    
    error_bvis_list = [arlexecute.execute(copy_visibility, nout=1)(bvis, zero=True) for bvis in sub_bvis_list]
    error_bvis_list = [predict_skymodel_list_compsonly_arlexecute_workflow(error_bvis_list[ibv],
                                                                           error_sm_list[ibv],
                                                                           context='2d', docal=True)
                       for ibv, bvis in enumerate(error_bvis_list)]
    
    # Inner nest is bvis per skymodels, outer is over vis's. Calculate residual visibility
    def subtract_vis_convert(error_bvis, no_error_bvis):
        error_bvis.data['vis'] = error_bvis.data['vis'] - no_error_bvis.data['vis']
        error_vis = convert_blockvisibility_to_visibility(error_bvis)
        return error_vis
    
    error_vis_list = [[arlexecute.execute(subtract_vis_convert)(error_bvis_list[ibvis][icomp],
                                                                no_error_bvis_list[ibvis][icomp])
                       for icomp, _ in enumerate(sub_components)]
                      for ibvis, _ in enumerate(error_bvis_list)]
    
    # Now for each visibility/component, we make the component dirty images. We just add these
    # component dirty images since the weights should be the same
    def sum_images(images):
        sum_image = create_empty_image_like(images[0][0])
        for im in images:
            sum_image.data += im[0].data
        return sum_image, images[0][1]
    
    dirty_list = list()
    for vis in error_vis_list:
        result = invert_list_arlexecute_workflow(vis, sub_model_list, '2d')
        dirty_list.append(arlexecute.execute(sum_images)(result))
    
    return dirty_list


if __name__ == '__main__':
    
    print(" ")
    print("Distributed simulation of surface errors for SKA-MID")
    print("----------------------------------------------------")
    print(" ")
    
    memory_use = dict()
    
    # Get command line inputs
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate pointing errors')
    parser.add_argument('--context', type=str, default='singlesource',
                        help='s3sky or singlesource')
    
    parser.add_argument('--rmax', type=float, default=1e5,
                        help='Maximum distance of station from centre (m)')
    
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--memory', type=int, default=8, help='Memory per worker (GB)')
    parser.add_argument('--nworkers', type=int, default=8, help='Number of workers')
    parser.add_argument('--flux_limit', type=float, default=1.0, help='Flux limit (Jy)')
    parser.add_argument('--show', type=str, default='True', help='Show images?')
    parser.add_argument('--export_images', type=str, default='True', help='Export images in fits format?')
    parser.add_argument('--ngroup_visibility', type=int, default=8, help='Process in visibility groups this large')
    parser.add_argument('--ngroup_components', type=int, default=8, help='Process in component groups this large')
    parser.add_argument('--npixel', type=int, default=512, help='Number of pixels in image')
    parser.add_argument('--seed', type=int, default=18051955, help='Random number seed')
    parser.add_argument('--snapshot', type=str, default='False', help='Do snapshot only?')
    parser.add_argument('--opposite', type=str, default='False',
                        help='Move source to opposite side of pointing centre')
    parser.add_argument('--pbradius', type=float, default=2.0, help='Radius of sources to include (in HWHM)')
    parser.add_argument('--pbtype', type=str, default='MID', help='Primary beam model: MID or MID_GAUSS')
    parser.add_argument('--use_agg', type=str, default="True", help='Use Agg matplotlib backend?')
    parser.add_argument('--declination', type=float, default=-45.0, help='Declination (degrees)')
    parser.add_argument('--tsys', type=float, default=0.0, help='System temperature: standard 20K')
    parser.add_argument('--scale', type=float, nargs=2, default=[1.0, 1.0], help='Scale errors by this amount')
    parser.add_argument('--use_radec', type=str, default="False", help='Calculate in RADEC (false)?')
    parser.add_argument('--use_natural', type=str, default="False", help='Use natural weighting?')
    parser.add_argument('--integration_time', type=float, default=600.0, help="Integration time (s)")
    parser.add_argument('--time_range', type=float, nargs=2, default=[-6.0, 6.0], help="Hourangle range (hours")
    parser.add_argument('--time_chunk', type=float, default=1800.0, help="Time for a chunk (s)")
    parser.add_argument('--shared_directory', type=str, default='../shared/',
                        help='Location of configuration files')
    parser.add_argument('--vp_directory', type=str, default='./',
                        help='Location of voltage pattern files')

    args = parser.parse_args()
    
    use_agg = args.use_agg == "True"
    if use_agg:
        import matplotlib as mpl
        
        mpl.use('Agg')
    from matplotlib import pyplot as plt
    
    declination = args.declination
    use_radec = args.use_radec == "True"
    use_natural = args.use_natural == "True"
    export_images = args.export_images == "True"
    tsys = args.tsys
    integration_time = args.integration_time
    time_range = args.time_range
    time_chunk = args.time_chunk
    snapshot = args.snapshot == 'True'
    opposite = args.opposite == 'True'
    pbtype = args.pbtype
    pbradius = args.pbradius
    rmax = args.rmax
    flux_limit = args.flux_limit
    npixel = args.npixel
    shared_directory = args.shared_directory
    vp_directory = args.vp_directory

    seed = args.seed
    print("Random number seed is", seed)
    show = args.show == 'True'
    context = args.context
    nworkers = args.nworkers
    nnodes = args.nnodes
    threads_per_worker = args.nthreads
    memory = args.memory
    ngroup_visibility = args.ngroup_visibility
    ngroup_components = args.ngroup_components
    
    basename = os.path.basename(os.getcwd())
    
    client = get_dask_Client(threads_per_worker=threads_per_worker,
                             processes=threads_per_worker == 1,
                             memory_limit=memory * 1024 * 1024 * 1024,
                             n_workers=nworkers)
    arlexecute.set_client(client=client)
    # n_workers is only relevant if we are using LocalCluster (i.e. a single node) otherwise
    # we need to read the actual number of workers
    time_started = time.time()
    
    # Set up details of simulated observation
    nfreqwin = 1
    diameter = 15.
    
    frequency = [1.4e9]
    channel_bandwidth = [1e7]
    
    # Do each 30 minutes in parallel
    start_times = numpy.arange(time_range[0] * 3600, time_range[1] * 3600, time_chunk)
    end_times = start_times + time_chunk
    print("Start times for chunks:")
    pp.pprint(start_times)
    
    times = [numpy.arange(start_times[itime], end_times[itime], integration_time) for itime in
             range(len(start_times))]
    print("Observation times:")
    s2r = numpy.pi / (12.0 * 3600)
    rtimes = s2r * numpy.array(times)
    ntimes = len(rtimes.flat)
    nchunks = len(start_times)
    
    print('%d integrations of duration %.1f s processed in %d chunks' % (ntimes, integration_time, nchunks))
    
    phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
    location = EarthLocation(lon="21.443803", lat="-30.712925", height=0.0)
    mid = create_configuration_from_MIDfile('%s/ska1mid_local.cfg' % shared_directory, rmax=rmax,
                                            location=location)
    
    bvis_graph = [arlexecute.execute(create_blockvisibility)(mid, rtimes[itime], frequency=frequency,
                                                             channel_bandwidth=channel_bandwidth, weight=1.0,
                                                             phasecentre=phasecentre,
                                                             polarisation_frame=PolarisationFrame("stokesI"),
                                                             zerow=True)
                  for itime in range(nchunks)]
    future_bvis_list = arlexecute.persist(bvis_graph, sync=True)
    
    bvis_list0 = arlexecute.compute(bvis_graph[0], sync=True)
    
    memory_use['bvis_list'] = nchunks * bvis_list0.size()
    
    vis_graph = [arlexecute.execute(convert_blockvisibility_to_visibility)(bv) for bv in future_bvis_list]
    future_vis_list = arlexecute.persist(vis_graph, sync=True)
    
    vis_list0 = arlexecute.compute(vis_graph[0], sync=True)
    memory_use['vis_list'] = nchunks * vis_list0.size()
    
    # We need the HWHM of the primary beam. Got this by trial and error
    if pbtype == 'MID':
        HWHM_deg = 0.596 * 1.4e9 / frequency[0]
    elif pbtype == 'MID_GRASP':
        HWHM_deg = 0.751 * 1.4e9 / frequency[0]
    elif pbtype == 'MID_GAUSS':
        HWHM_deg = 0.766 * 1.4e9 / frequency[0]
    else:
        HWHM_deg = 0.596 * 1.4e9 / frequency[0]
    
    HWHM = HWHM_deg * numpy.pi / 180.0
    
    FOV_deg = 5.0 * HWHM_deg
    print('%s: HWHM beam = %g deg' % (pbtype, HWHM_deg))
    
    advice_list = arlexecute.execute(advise_wide_field)(future_vis_list[0], guard_band_image=1.0,
                                                        delA=0.02)
    advice = arlexecute.compute(advice_list, sync=True)
    pb_npixel = 1024
    d2r = numpy.pi / 180.0
    pb_cellsize = d2r * FOV_deg / pb_npixel
    cellsize = advice['cellsize']
    
    if show:
        future_vis_list = arlexecute.compute(future_vis_list, sync=True)
        plt.clf()
        for ivis in range(nchunks):
            vis = future_vis_list[ivis]
            plt.plot(-vis.u, -vis.v, '.', color='b', markersize=0.2)
            plt.plot(vis.u, vis.v, '.', color='b', markersize=0.2)
        plt.xlabel('U (wavelengths)')
        plt.ylabel('V (wavelengths)')
        plt.title('UV coverage')
        plt.savefig('uvcoverage.png')
        plt.show(block=False)
        future_vis_list = arlexecute.scatter(future_vis_list)
        
        plt.clf()
        r2d = 180.0 / numpy.pi
        future_bvis_list = arlexecute.compute(future_bvis_list, sync=True)
        for ivis in range(nchunks):
            bvis = future_bvis_list[ivis]
            ha = numpy.pi * bvis.time / 43200.0
            dec = phasecentre.dec.rad
            latitude = bvis.configuration.location.lat.rad
            az, el = hadec_to_azel(ha, dec, latitude)
            if ivis == 0:
                plt.plot(ha, r2d * az, '.', color='r', label='Azimuth (deg)')
                plt.plot(ha, r2d * el, '.', color='b', label='Elevation (deg)')
            else:
                plt.plot(ha, r2d * az, '.', color='r')
                plt.plot(ha, r2d * el, '.', color='b')
        plt.xlabel('HA (s)')
        plt.ylabel('Angle')
        plt.legend()
        plt.title('Azimuth and elevation vs hour angle')
        plt.savefig('azel.png')
        plt.show(block=False)
        future_bvis_list = arlexecute.scatter(future_bvis_list)
    
    # Construct the skycomponents
    if context == 'singlesource':
        print("Constructing single component")
        offset = [HWHM_deg, 0.0]
        if opposite:
            offset = [-1.0 * offset[0], -1.0 * offset[1]]
        print("Offset from pointing centre = %.3f, %.3f deg" % (offset[0], offset[1]))
        
        # The point source is offset to approximately the halfpower point
        offset_direction = SkyCoord(ra=(+15.0 + offset[0]) * u.deg,
                                    dec=(declination + offset[1]) * u.deg,
                                    frame='icrs', equinox='J2000')
        
        original_components = [Skycomponent(flux=[[1.0]], direction=offset_direction, frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesI'))]
        print(original_components[0])
    
    else:
        # Make a skymodel from S3
        max_flux = 0.0
        total_flux = 0.0
        print("Constructing s3sky components")
        from wrappers.serial.simulation.testing_support import create_test_skycomponents_from_s3
        
        original_components = create_test_skycomponents_from_s3(flux_limit=flux_limit / 100.0,
                                                                phasecentre=phasecentre,
                                                                polarisation_frame=PolarisationFrame("stokesI"),
                                                                frequency=numpy.array(frequency),
                                                                radius=pbradius * HWHM)
        print("%d components before application of primary beam" %
              (len(original_components)))
        
        pbmodel = arlexecute.execute(create_image_from_visibility)(vis_list0, npixel=pb_npixel,
                                                                   cellsize=pb_cellsize,
                                                                   override_cellsize=False,
                                                                   phasecentre=phasecentre,
                                                                   frequency=frequency,
                                                                   nchan=nfreqwin,
                                                                   polarisation_frame=PolarisationFrame(
                                                                       "stokesI"))
        pbmodel = arlexecute.compute(pbmodel, sync=True)
        # Use MID_GAUSS to filter the components since MID_GRASP is in local coordinates
        pb = create_pb(pbmodel, "MID_GAUSS", pointingcentre=phasecentre, use_local=False)
        pb_applied_components = [copy_skycomponent(c) for c in original_components]
        pb_applied_components = apply_beam_to_skycomponent(pb_applied_components, pb)
        filtered_components = []
        for icomp, comp in enumerate(pb_applied_components):
            if comp.flux[0, 0] > flux_limit:
                total_flux += comp.flux[0, 0]
                if abs(comp.flux[0, 0]) > max_flux:
                    max_flux = abs(comp.flux[0, 0])
                filtered_components.append(original_components[icomp])
        print("%d components > %.3f Jy after application of primary beam" %
              (len(filtered_components), flux_limit))
        print("Strongest components is %g (Jy)" % max_flux)
        print("Total flux in components is %g (Jy)" % total_flux)
        original_components = [copy_skycomponent(c) for c in filtered_components]
        plt.clf()
        show_image(pb, components=original_components)
        plt.show(block=False)
        
        print("Created %d components" % len(original_components))
        # Primary beam points to the phasecentre
        offset_direction = SkyCoord(ra=+15.0 * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
    
    ses = [0.0]
    
    nants = len(mid.names)
    nbaselines = nants * (nants - 1) // 2
    
    memory_use['model_list'] = 8 * npixel * npixel * len(frequency) * len(original_components) / 1024 / 1024 / 1024
    memory_use['vp_list'] = 16 * npixel * npixel * len(frequency) * nchunks / 1024 / 1024 / 1024
    print("Memory use (GB)")
    pp.pprint(memory_use)
    total_memory_use = numpy.sum([memory_use[key] for key in memory_use.keys()])
    
    print("Summary of processing:")
    print("    There are %d workers" % nworkers)
    print("    There are %d separate visibility time chunks being processed" % len(future_vis_list))
    print("    The integration time within each chunk is %.1f (s)" % integration_time)
    print("    There are a total of %d integrations" % ntimes)
    print("    There are %d baselines" % nbaselines)
    print("    There are %d components" % len(original_components))
    print("    %d surface scenario(s) will be tested" % len(ses))
    ntotal = ntimes * nbaselines * len(original_components) * len(ses)
    print("    Total processing %g times-baselines-components-scenarios" % ntotal)
    print("    Approximate total memory use for data = %.3f GB" % total_memory_use)
    nworkers = len(arlexecute.client.scheduler_info()['workers'])
    print("    Using %s Dask workers" % nworkers)
    
    # Uniform weighting
    future_model_list = [arlexecute.execute(create_image_from_visibility)(vis_list0, npixel=npixel,
                                                                          frequency=frequency,
                                                                          nchan=nfreqwin, cellsize=cellsize,
                                                                          phasecentre=offset_direction,
                                                                          polarisation_frame=PolarisationFrame(
                                                                              "stokesI"))
                         for i, _ in enumerate(original_components)]
    future_model_list = arlexecute.persist(future_model_list)
    
    psf_list = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, frequency=frequency,
                                                                 nchan=nfreqwin, cellsize=cellsize,
                                                                 phasecentre=phasecentre,
                                                                 polarisation_frame=PolarisationFrame("stokesI"))
                for v in future_vis_list]
    psf_list = arlexecute.compute(psf_list, sync=True)
    future_psf_list = arlexecute.scatter(psf_list)
    del psf_list
    
    if use_natural:
        print("Using natural weighting")
    else:
        print("Using uniform weighting")
        
        vis_list = weight_list_arlexecute_workflow(future_vis_list, future_psf_list)
        vis_list = arlexecute.compute(vis_list, sync=True)
        future_vis_list = arlexecute.scatter(vis_list)
        del vis_list
        
        bvis_list = [arlexecute.execute(convert_visibility_to_blockvisibility)(vis) for vis in future_vis_list]
        bvis_list = arlexecute.compute(bvis_list, sync=True)
        future_bvis_list = arlexecute.scatter(bvis_list)
    
    print("Inverting to get PSF")
    psf_list = invert_list_arlexecute_workflow(future_vis_list, future_psf_list, '2d', dopsf=True)
    psf_list = arlexecute.compute(psf_list, sync=True)
    psf, sumwt = sum_invert_results(psf_list)
    print("PSF sumwt ", sumwt)
    if export_images:
        export_image_to_fits(psf, 'PSF_arl.fits')
    if show:
        show_image(psf, cm='gray_r', title='%s PSF' % basename, vmin=-0.01, vmax=0.1)
        plt.savefig('PSF_arl.png')
        plt.show(block=False)
    del psf_list
    del future_psf_list
    
    # ### Calculate the voltage pattern without pointing errors
    vp = create_image_from_visibility(vis_list0, npixel=pb_npixel, frequency=frequency,
                                      nchan=nfreqwin, cellsize=pb_cellsize,
                                      phasecentre=phasecentre,
                                      override_cellsize=False)
    # Optionally show the primary beam, with components if the image is in RADEC coords
    if show:
        if pbtype == "MID_GRASP":
            pb = create_pb(vp, "MID_GAUSS", pointingcentre=phasecentre,
                           use_local=False)
        else:
            pb = create_pb(vp, pbtype, pointingcentre=phasecentre,
                           use_local=False)
        
        print("Primary beam:", pb)
        show_image(pb, title='%s: primary beam' % basename, components=original_components, vmax=1.0, vmin=0.0)
        
        plt.savefig('PB_arl.png')
        plt.show(block=False)
        if export_images:
            export_image_to_fits(pb, 'PB_arl.fits')
    
    # Construct the voltage patterns
    results = []
    
    filename = seqfile.findNextFile(prefix='surface_simulation_%s_' % socket.gethostname(), suffix='.csv')
    print('Saving results to %s' % filename)
    plotfile = seqfile.findNextFile(prefix='surface_simulation_%s_' % socket.gethostname(), suffix='.jpg')
    
    epoch = time.strftime("%Y-%m-%d %H:%M:%S")
    
    time_started = time.time()
    
    # Now loop over all surface errors
    print("")
    print("***** Starting loop over scenarios ******")
    print("")
    for se in ses:
        
        result = dict()
        result['context'] = context
        result['nb_name'] = sys.argv[0]
        result['plotfile'] = plotfile
        result['hostname'] = socket.gethostname()
        result['epoch'] = epoch
        result['basename'] = basename
        result['nworkers'] = nworkers
        result['ngroup_visibity'] = ngroup_visibility
        result['ngroup_components'] = ngroup_components
        
        result['npixel'] = npixel
        result['pb_npixel'] = pb_npixel
        result['flux_limit'] = flux_limit
        result['pbtype'] = pbtype
        result['surface_scaling'] = se
        result['snapshot'] = snapshot
        result['opposite'] = opposite
        result['tsys'] = tsys
        result['declination'] = declination
        result['use_radec'] = use_radec
        result['use_natural'] = use_natural
        result['integration_time'] = integration_time
        result['seed'] = seed
        result['ntotal'] = ntotal
        result['se'] = se
        
        a2r = numpy.pi / (3600.0 * 180.0)
        
        # The strategy for distribution is to iterate through big cells in (bvis, components). Within each
        # cell we do distribution over the each bvis, component using create_vis_list_with_error
        chunk_components = [original_components[i:i + ngroup_components]
                            for i in range(0, len(original_components), ngroup_components)]
        chunk_bvis = [future_bvis_list[i:i + ngroup_visibility]
                      for i in range(0, len(future_bvis_list), ngroup_visibility)]
        
        file_name = 'SE_type1_%.3f_%s' % (se, context)
        
        error_dirty_list = list()
        for icomp_chunk, comp_chunk in enumerate(chunk_components):
            for ivis_chunk, vis_chunk in enumerate(chunk_bvis):
                print("Processing component_chunk %d, visibility chunk %d" % (icomp_chunk, ivis_chunk))
                vis_comp_chunk_dirty_list = create_vis_list_with_errors(chunk_bvis[ivis_chunk],
                                                                        chunk_components[icomp_chunk],
                                                                        sub_model_list=future_model_list,
                                                                        vp_directory=vp_directory,
                                                                        use_radec=use_radec)
                this_result = arlexecute.compute(vis_comp_chunk_dirty_list, sync=True)
                for r in this_result:
                    error_dirty_list.append(r)
        
        error_dirty, sumwt = sum_invert_results(error_dirty_list)
        
        
        print("Dirty image sumwt", sumwt)
        del error_dirty_list
        print(qa_image(error_dirty))
        
        if show:
            show_image(error_dirty, cm='gray_r')
            plt.savefig('%s.png' % file_name)
            plt.show(block=False)
        
        qa = qa_image(error_dirty)
        _, _, ny, nx = error_dirty.shape
        for field in ['maxabs', 'rms', 'medianabs']:
            result["onsource_" + field] = qa.data[field]
        result['onsource_abscentral'] = numpy.abs(error_dirty.data[0, 0, ny // 2, nx // 2])
        
        qa_psf = qa_image(psf)
        _, _, ny, nx = psf.shape
        for field in ['maxabs', 'rms', 'medianabs']:
            result["psf_" + field] = qa_psf.data[field]
        
        result['elapsed_time'] = time.time() - time_started
        print('Elapsed time = %.1f (s)' % result['elapsed_time'])
        
        results.append(result)
    
    pp.pprint(results)
    
    print("Total processing %g times-baselines-components-scenarios" % ntotal)
    processing_rate = ntotal / (nworkers * (time.time() - time_started))
    print("Processing rate of time-baseline-component-scenario = %g per worker-second" % processing_rate)
    
    for result in results:
        result["processing_rate"] = processing_rate
    
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys(), delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        csvfile.close()