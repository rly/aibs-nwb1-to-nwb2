"""Create an NWB 2 file that merges the contents of an NWB 1.0 file and an NWB 2.0 file that is output from suite2p.

Note that we do not simply append to the NWB 2.0 file that is output from suite2p because
once an NWB file is created, some properties cannot be changed, such as the identifier, session_description,
file_create_date, and session_start_time. In order to preserve these properties, we create a new NWB 2.0 file
that contains the data from the NWB 1.0 file and the suite2p output NWB 2.0 file.
"""
from datetime import datetime, timezone
import h5py
import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO, NWBFile, TimeSeries  # , H5DataIO

# from pynwb.base import Images
from pynwb.behavior import BehavioralTimeSeries
from pynwb.file import Subject
from pynwb.image import OpticalSeries, IndexSeries

# from pynwb.image import GrayscaleImage
from pynwb.misc import IntervalSeries
import pytz
import argparse
import os

def main(path_nwb_1, path_nwb_2, path_output):
    """Main function."""
    old_nwb_path = path_nwb_1
    suite2p_out_path = path_nwb_2
    temp_merged_path = os.path.join(os.path.split(path_output)[0], 'temp.nwb')
    export_path = path_output

    # open the suite2p output NWB file in read mode
    with NWBHDF5IO(suite2p_out_path, "r") as io:
        in_nwbfile = io.read()

        # open the NWB 1 file using h5py
        # this code assumes that relevant data lives in particular places and only those places
        with h5py.File(old_nwb_path) as f:
            out_nwbfile = create_out_nwbfile(f, in_nwbfile)
            add_running_speed_timeseries(out_nwbfile, f)
            # TODO add pupil tracking
            # TODO add eye tracking
            add_stimuli(out_nwbfile, f)
            add_subject(out_nwbfile, f)
            add_general(out_nwbfile, f)

            # TODO add imaging plane
            # TODO how is the imaging plane in NWB 1 file different from the one in suite2p file?
            # NOTE the raw acquisition data is omitted

        # NOTE in order to add elements from the NWB 2 file into a new file, the new file must
        # first be created, then read, then have elements added to it, then exported to a new file.
        # this is not the most efficient, but it works.
        with NWBHDF5IO(temp_merged_path, "w") as merged_io_write:
            merged_io_write.write(out_nwbfile)

        with NWBHDF5IO(temp_merged_path, "r") as merged_io_read:
            export_nwbfile = merged_io_read.read()

            # TODO fix transpose of data in suite2p output
            # see https://github.com/MouseLand/suite2p/issues/909
            add_suite2p_output(export_nwbfile, in_nwbfile)

            with NWBHDF5IO(export_path, "w") as export_merged_io:
                export_merged_io.export(
                    src_io=merged_io_read,
                    nwbfile=export_nwbfile,
                    write_args={"link_data": False},
                )
    
    # we remove tmp file
    os.remove(temp_merged_path)

def _unicode(s: str | bytes):
    """A helper function for converting a string or bytes object to Unicode."""
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode("utf-8")
    else:
        raise ValueError("Expected unicode or ascii string, got %s" % type(s))


def create_out_nwbfile(f: h5py.File, in_nwbfile: NWBFile) -> NWBFile:
    """Create a new NWBFile with the same base properties as the NWB 1 file."""
    # convert from "Tue Jan 26 12:28:49 2016" format to datetime object
    strptime_format = r"%a %b %d %H:%M:%S %Y"
    in_nwbfile_tz = pytz.timezone("US/Pacific")  # Allen Institute timezone

    # artificially store history of when data were generated
    old_file_create_date = datetime.strptime(
        _unicode(f["/file_create_date"][0]), strptime_format
    )
    old_file_create_date = in_nwbfile_tz.localize(old_file_create_date)
    file_create_date = [
        old_file_create_date,
        in_nwbfile.file_create_date[0],
        datetime.now(timezone.utc).astimezone(),
    ]

    old_session_start_time = datetime.strptime(
        _unicode(f["/session_start_time"][()]), strptime_format
    )
    old_session_start_time = in_nwbfile_tz.localize(old_session_start_time)

    # pynwb does not allow these values to be reset so create a new nwbfile
    out_nwbfile = NWBFile(
        identifier=_unicode(f["/identifier"][()]),
        session_description=_unicode(f["/session_description"][()]),
        file_create_date=file_create_date,
        session_start_time=old_session_start_time,
    )
    return out_nwbfile


def add_running_speed_timeseries(out_nwbfile: NWBFile, f: h5py.File):
    """Add running speed data from the NWB 1 file to the suite2p output NWB file."""
    old_running_speed = f[
        "/processing/brain_observatory_pipeline/BehavioralTimeSeries/running_speed"
    ]
    ts = TimeSeries(
        name="running_speed",
        data=old_running_speed["data"][:],
        timestamps=old_running_speed["timestamps"][:],
        unit="frame",
        description=_unicode(old_running_speed.attrs["description"]),
        comments=_unicode(old_running_speed.attrs["comments"]),
    )

    behavioral_timeseries = BehavioralTimeSeries()
    behavioral_timeseries.add_timeseries(ts)
    behavior_module = out_nwbfile.create_processing_module(
        name="behavior", description="processed behavioral data"
    )
    behavior_module.add(behavioral_timeseries)


def add_stimuli(out_nwbfile: NWBFile, f: h5py.File):
    """Add presented stimuli from the NWB 1 file to the suite2p output NWB file."""

    def add_stimulus(template_name: str, presentation_name: str):
        # using an IndexSeries on a TimeSeries will result in a PendingDeprecationWarning from pynwb
        # however, the recommended method does not currently work due to a bug in pynwb
        # so we use this instead

        # NOTE: using Images will omit the information in the "dimension" and "field_of_view" datasets
        # from the NWB 1 file, so it may be better to stick with this method even if using Images is
        # recommended instead of an OpticalSeries

        # old_lsn_data = old_lsn_template["data"][:]  # num_images x num_x x num_y
        # lsn_stimulus_images = Images(
        #     name="locally_sparse_noise_stimulus",
        # )
        # for i in range(old_lsn_data.shape[0]):
        #     image = GrayscaleImage(
        #         name=str(i),
        #         data=np.flatten(old_lsn_data[i,:,:]),
        #     )
        #     lsn_stimulus_images.add_image(image)
        # in_nwbfile.add_stimulus_template(lsn_stimulus_images)

        old_template = f[f"/stimulus/templates/{template_name}"]
        new_template = OpticalSeries(
            name=Path(old_template.name).name,
            data=old_template["data"][:],
            dimension=old_template["dimension"][:],
            field_of_view=old_template["field_of_view"][:],
            format=_unicode(old_template["format"][()]),
            starting_time=0.0,  # time is meaningless here
            rate=0.0,  # time is meaningless here
            description=_unicode(old_template.attrs["description"][:]),
            comments=_unicode(old_template.attrs["comments"][:]),
            distance=-1.0,  # placeholder
            orientation="N/A",  # placeholder
            unit="N/A",
        )
        out_nwbfile.add_stimulus_template(new_template)

        old_presentation = f[f"/stimulus/presentation/{presentation_name}"]
        new_presentation = IndexSeries(
            name=Path(old_presentation.name).name,
            data=old_presentation["data"][:].astype(np.uint32),
            timestamps=old_presentation["timestamps"][:],
            indexed_timeseries=new_template,
            description=_unicode(old_presentation.attrs["description"]),
            comments=_unicode(old_presentation.attrs["comments"]),
            unit="N/A",
        )
        out_nwbfile.add_stimulus(new_presentation)

        # NOTE the NWB 1 file contains frame_duration, which appears to be N x 2 representing the start_frame index and
        # stop_frame index for each presentation. this data does not have a corresponding place in the NWB 2 core
        # schema. this is currently omitted from the conversion. an extension could be written to include this data.

    add_stimulus(
        template_name="locally_sparse_noise_image_stack",
        presentation_name="locally_sparse_noise_stimulus",
    )
    add_stimulus(
        template_name="natural_movie_one_image_stack",
        presentation_name="natural_movie_one_stimulus",
    )
    add_stimulus(
        template_name="natural_movie_two_image_stack",
        presentation_name="natural_movie_two_stimulus",
    )

    # spontaneous stimulus has no corresponding image stack
    old_spont_presentation = f["/stimulus/presentation/spontaneous_stimulus"]
    new_spont_presentation = IntervalSeries(
        name="spontaneous_stimulus",
        data=old_spont_presentation["data"][:],
        timestamps=old_spont_presentation["timestamps"][:],
        description=_unicode(old_spont_presentation.attrs["description"][:]),
        comments=_unicode(old_spont_presentation.attrs["comments"][:]),
    )
    out_nwbfile.add_stimulus(new_spont_presentation)
    # NOTE the NWB 1 file contains frame_duration, which appears to be N x 2 representing the start_frame index and
    # stop_frame index for each presentation. this data does not have a corresponding place in the NWB 2 core schema.
    # this is currently omitted from the conversion. an extension could be written to include this data.
    # also note that for spontaneous_stimulus, start_frame == end_frame which seems incorrect


def add_subject(in_nwbfile: NWBFile, f: h5py.File):
    """Add subject information from the NWB 1 file to the suite2p output file."""
    old_subject = f["/general/subject"]

    # change the value of sex to meet NWB 2 best practices
    sex = _unicode(old_subject["sex"][()])
    if sex == "male":
        new_sex = "M"
    elif sex == "female":
        new_sex = "F"
    else:
        raise ValueError(f"Unexpected value for subject 'sex' in NWB 1 file: {sex}")

    new_subject = Subject(
        age=_unicode(old_subject["age"][()]),
        description=_unicode(old_subject["description"][()]),
        genotype=_unicode(old_subject["genotype"][()]),
        sex=new_sex,
        species=_unicode(old_subject["species"][()]),
        subject_id=_unicode(old_subject["subject_id"][()]),
    )
    in_nwbfile.subject = new_subject


def add_general(out_nwbfile: NWBFile, f: h5py.File):
    """Add general metadata from the NWB 1 file to the suite2p output file."""
    out_nwbfile.institution = _unicode(f["/general/institution"][()])
    out_nwbfile.session_id = _unicode(f["/general/session_id"][()])
    # NOTE for session id, there is a comment saying "ID corresponds to Allen Institute 'experiment_sessions ID'""

    # NOTE the NWB 1 file also contains the following datasets in "/general":
    # For more information
    # experiment_container_id
    # fov
    # generated_by
    # ophys_experiment_id
    # ophys_experiment_name
    # pixel_size
    # session_type
    # specimen_name
    # targeted_structure
    #
    # these data do not have a corresponding place in the NWB 2 core schema.
    # these are currently omitted from the conversion. an extension could be written to include these data.

    for device_name in f["/general/devices"].keys():
        out_nwbfile.create_device(name=device_name)


def add_suite2p_output(out_nwbfile: NWBFile, in_nwbfile: NWBFile):
    """Copy the suite2p output data to the new NWB file."""
    in_nwbfile.processing["ophys"].reset_parent()
    out_nwbfile.add_processing_module(in_nwbfile.processing["ophys"])

    in_nwbfile.acquisition["TwoPhotonSeries"].reset_parent()
    out_nwbfile.add_acquisition(in_nwbfile.acquisition["TwoPhotonSeries"])

    in_nwbfile.imaging_planes["ImagingPlane"].reset_parent()
    out_nwbfile.add_imaging_plane(in_nwbfile.imaging_planes["ImagingPlane"])

    # the suite2p output includes a dummy 2p microscope device. instead of using that one,
    # use the original device. but the device was already set, so we need to bypass the
    # pynwb restriction
    out_nwbfile.imaging_planes["ImagingPlane"].fields.pop("device")
    # NOTE the following requires HDMF 3.4.8 to create the correct link
    out_nwbfile.imaging_planes["ImagingPlane"].device = out_nwbfile.devices[
        "2-photon microscope"
    ]

    # remove the reference images from the PlaneSegmentation.
    # they have no useful data and will result in a broken link
    out_nwbfile.processing["ophys"]["ImageSegmentation"][
        "PlaneSegmentation"
    ].reference_images.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_nwb_1', type=str, required=True)
    parser.add_argument('--path_nwb_2', type=str, required=True)
    parser.add_argument('--path_output', type=str, required=True)

    args = parser.parse_args()
    path_nwb_1 = args.path_nwb_1
    path_nwb_2 = args.path_nwb_2
    path_output = args.path_output


    main(path_nwb_1, path_nwb_2, path_output)
