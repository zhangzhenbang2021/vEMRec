import os
import numpy as np
from struct import pack, unpack
import time

def readFlowFile(fname):
    '''
    args
        fname (str)
    return
        flow (numpy array) numpy array of shape (height, width, 2)
    '''

    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('readFlowFile: fname %s should have extension ''.flo''' % fname)

    try:
        fid = open(fname, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', fname)

    tag     = unpack('f', fid.read(4))[0]
    width   = unpack('i', fid.read(4))[0]
    height  = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % fname)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (fname, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (fname, height))

    nBands = 2

    # arrange into matrix form
    flow = np.fromfile(fid, np.float32)
    flow = flow.reshape(height, width, nBands)

    fid.close()

    return flow

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d

def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow


def writeFlowFile(img, fname):
    TAG_STRING = 'PIEH'    # use this when WRITING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('writeFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('writeFlowFile: fname %s should have extension ''.flo''', fname)

    height, width, nBands = img.shape

    assert nBands == 2, 'writeFlowFile: image must have two bands'


    fid = open(fname, 'wb')
    fid.write(bytes(TAG_STRING, 'utf-8'))
    fid.write(pack('i', width))
    fid.write(pack('i', height))
    tmp = np.zeros((height, width*nBands), np.float32)
    tmp[:, np.arange(width) * nBands] = img[:, :, 0]
    tmp[:, np.arange(width) * nBands + 1] = np.squeeze(img[:, :, 1])
    fid.write(bytes(tmp))
    fid.close()

def write_flow(flow, filename, max_retries=10):
    """
    Write optical flow in Middlebury .flo format.
    
    Parameters:
    - flow: optical flow map
    - filename: optical flow file path to be saved
    - max_retries: maximum number of retries on error (default: 3)
    
    Returns: None
    """
    retries = 0
    while retries < max_retries:
        try:
            with open(filename, 'wb') as f:
                magic = np.array([202021.25], dtype=np.float32)
                (height, width) = flow.shape[0:2]
                w = np.array([width], dtype=np.int32)
                h = np.array([height], dtype=np.int32)
                magic.tofile(f)
                w.tofile(f)
                h.tofile(f)
                flow.tofile(f)
            return  # 成功写入，退出函数
        except OSError as e:
            print(f"Error writing to file {filename}: {e}")
            retries += 1
            time.sleep(1)  # 等待一段时间后重试
    
    # 如果达到最大重试次数仍无法写入，则打印错误信息并可能抛出异常
    print(f"Failed to write to file {filename} after {max_retries} retries.")
    raise RuntimeError(f"Failed to write to file {filename} after {max_retries} retries.")



def compare_arrays(arr1, arr2, threshold=1e-6):
    if arr1.shape != arr2.shape:
        return False
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    return max_diff < threshold

def modify_file_path(flow_path,i):
    base_path, extension = os.path.splitext(flow_path)
    new_flow_path = base_path + f'_{i}' + extension
    return new_flow_path

def flow_save(flow,in_path):
    ''' flow:torch B2HW cpu
    '''
    i=0
    raw_path = in_path
    flow_path = modify_file_path(raw_path,i)
    if os.path.exists(flow_path):
        while os.path.exists(flow_path):
            i+=1
            flow_path = modify_file_path(raw_path,i)
            
    flow = flow.permute(0,2,3,1) # BHW2
    flow = flow.squeeze(0).data.numpy() #HW2
    
    write_flow(flow,flow_path)
    
    
    