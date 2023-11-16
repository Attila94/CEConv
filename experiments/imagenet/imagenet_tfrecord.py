from subprocess import call
import os

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def

    import nvidia.dali.fn as fn

    import nvidia.dali.types as types
    import nvidia.dali.tfrecord as tfrec
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


@pipeline_def
def tfrecord_pipeline(
    tfrecord_path,
    tfrecord_idx_path,
    num_gpus,
    is_training,
    shard_id,
    dali_cpu=False,
    crop=224,
    size=256,
    jitter=0.0,
    grayscale=False,
):
    # Specify devices to use.
    dali_device = "cpu" if dali_cpu else "gpu"
    decoder_device = "cpu" if dali_cpu else "mixed"
    # This padding sets the size of the internal nvJPEG buffers to be able to
    # handle all images from full-sized ImageNet without additional reallocations.
    device_memory_padding = 211025920 if decoder_device == "mixed" else 0
    host_memory_padding = 140544512 if decoder_device == "mixed" else 0

    inputs = fn.readers.tfrecord(  # type: ignore
        path=tfrecord_path,
        index_path=tfrecord_idx_path,
        features={
            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),  # type: ignore
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),  # type: ignore
        },
        num_shards=num_gpus,
        shard_id=shard_id,
        name="Reader",
    )

    # Decoder and data augmentation
    if is_training:
        images = fn.decoders.image_random_crop(  # type: ignore
            inputs["image/encoded"],
            device=decoder_device,
            output_type=types.RGB,  # type: ignore
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100,
        )
        images = fn.resize(  # type: ignore
            images,
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_TRIANGULAR,  # type: ignore
        )
        rng = fn.random.coin_flip()  # type: ignore
    else:
        images = fn.decoders.image(  # type: ignore
            inputs["image/encoded"],
            device=decoder_device,
            output_type=types.RGB,  # type: ignore
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        images = fn.resize(  # type: ignore
            images,
            device=dali_device,
            resize_shorter=size,
            interp_type=types.INTERP_TRIANGULAR,  # type: ignore
        )
        rng = False

    # Convert to grayscale.
    if grayscale:
        images = fn.saturation(images.gpu(), saturation=0.0)  # type: ignore

    # Apply hue transformation to each sample in the batch.
    if jitter > 0.0:
        jitter_factor = fn.random.uniform(range=(-360 * jitter, 360 * jitter))  # type: ignore
        images = fn.hue(images.gpu(), hue=jitter_factor, device=dali_device)  # type: ignore

    # Normalize such that values are in the range [0, 1].
    mean = [0.0, 0.0, 0.0]
    std = [255.0, 255.0, 255.0]

    images = fn.crop_mirror_normalize(  # type: ignore
        images.gpu(),
        dtype=types.FLOAT,  # type: ignore
        output_layout=types.NCHW,  # type: ignore
        crop=(crop, crop),
        mean=mean,
        std=std,
        mirror=rng,
    )

    labels = inputs["image/class/label"] - 1

    return images, labels


def ImageNet_TFRecord(
    root: str,
    split: str,
    batch_size: int,
    num_threads: int,
    device_id: int,
    num_gpus: int,
    dali_cpu: bool = False,
    is_training: bool = False,
    jitter: float = 0.0,
    grayscale: bool = False,
    subset: float = 1.0,
):
    """
    PyTorch dataloader for ImageNet TFRecord files.

    Args:
        root (str): Location of the 'tfrecords' ImageNet directory.
        split (str): Split to use, either 'train' or 'val'.
        batch_size (int): Batch size per GPU (default=64).
        num_threads (int): Number of dataloader workers to use per sub-process.
        device_id (int): ID of the GPU corresponding to the current subprocess. Dataset
            will be divided over all subprocesses.
        num_gpus (int): Total number of GPUS available.
        dali_cpu (bool): Set True to perform part of data loading on CPU instead of GPU (default=False).
        is_training (bool): Set True to use training preprocessing (default=False).
        jitter (float): Set to a value between 0 and 0.5 to apply random hue jitter to images (default=0.0).
        grayscale (bool): Set True to convert images to grayscale (default=False).
        subset (float): Fraction of dataset to use (default=1.0).

    Returns:
        PyTorch dataloader.

    """

    # List all tfrecord files in directory.
    tf_files = sorted(os.listdir(os.path.join(root, split, "data")))

    # Take subset of tfrecord files if subset < 1.0.
    if subset < 1.0:
        tf_files = tf_files[: int(len(tf_files) * subset)]

    # Create dir for idx files if not exists.
    idx_files_dir = os.path.join(root, split, "idx_files")
    if not os.path.exists(idx_files_dir):
        os.mkdir(idx_files_dir)

    tfrec_path_list = []
    idx_path_list = []
    n_samples = 0
    # Create idx files and create TFRecordPipelines.
    for tf_file in tf_files:
        # Path of tf_file and idx file.
        tfrec_path = os.path.join(root, split, "data", tf_file)
        tfrec_path_list.append(tfrec_path)
        idx_path = os.path.join(idx_files_dir, tf_file + "_idx")
        idx_path_list.append(idx_path)
        # Create idx file for tf_file by calling tfrecord2idx script.
        if not os.path.isfile(idx_path):
            call(["tfrecord2idx", tfrec_path, idx_path])
        with open(idx_path, "r") as f:
            n_samples += len(f.readlines())
    # Create TFRecordPipeline for each TFRecord file.
    pipe = tfrecord_pipeline(
        tfrec_path_list,
        idx_path_list,
        is_training=is_training,
        device_id=device_id,  # type: ignore
        shard_id=device_id,
        num_gpus=num_gpus,
        batch_size=batch_size,  # type: ignore
        num_threads=num_threads,  # type: ignore
        dali_cpu=dali_cpu,
        jitter=jitter,
        grayscale=grayscale,
    )
    pipe.build()

    dataloader = DALIClassificationIterator(
        pipelines=pipe,
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )
    return dataloader


if __name__ == "__main__":
    # Create dataloader.
    dataloader = ImageNet_TFRecord(
        root="/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/imagenet/tfrecords",
        split="train",
        batch_size=64,
        num_threads=2,
        device_id=0,
        num_gpus=1,
        dali_cpu=False,
        is_training=True,
        jitter=0.5,
        grayscale=False,
        subset=0.1,
    )

    # Get first batch and print shape.
    print("Number of batches: ", len(dataloader))
    data = next(iter(dataloader))
    print(data[0]["data"].shape, data[0]["label"].shape)  # type: ignore
