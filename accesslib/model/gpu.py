""" GPU configuration """
import tensorflow as tf
from tensorflow.python.client import device_lib


def _sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def print_devices():
    devices = device_lib.list_local_devices()
    for d in devices:
        t = d.device_type
        name = d.physical_device_desc
        _l = [item.split(':', 1) for item in name.split(", ")]
        name_attr = dict([x for x in _l if len(x) == 2])
        dev = name_attr.get('name', 'Unnamed device')
        print(f" {d.name} || {dev} || {t} || {_sizeof_fmt(d.memory_limit)}")


def configure_gpu_memory_allocation(memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 10GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU available.")


__all__ = [
    "print_devices",
    "configure_gpu_memory_allocation"
]
