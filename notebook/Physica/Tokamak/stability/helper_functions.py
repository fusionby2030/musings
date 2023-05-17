from typing import Dict, List, Tuple
import swiftclient
from io import BytesIO
import numpy as np 
import os 
import torch 
def get_allas_connection() -> swiftclient.Connection: 
    _authurl = os.environ['OS_AUTH_URL']
    _auth_version = os.environ['OS_IDENTITY_API_VERSION']
    _user = os.environ['OS_USERNAME']
    _key = os.environ['OS_PASSWORD']
    _os_options = {
        'user_domain_name': os.environ['OS_USER_DOMAIN_NAME'],
        'project_domain_name': os.environ['OS_USER_DOMAIN_NAME'],
        'project_name': os.environ['OS_PROJECT_NAME']
    }

    conn = swiftclient.Connection(
        authurl=_authurl,
        user=_user,
        key=_key,
        os_options=_os_options,
        auth_version=_auth_version
    )
    return conn  

from typing import Union, List, Dict, Optional, NewType
import pickle 
import aug_sfutils

PULSE_DICT = NewType('AUG_DICT', Dict[str, Dict[str, Dict[str, np.ndarray]]]) 
def get_pulse_dict(shot_number: Union[int, str], conn: swiftclient.Connection, bucket_name: str = 'AUG_PULSES') -> Optional[PULSE_DICT]: 
    try: 
        my_obj = conn.get_object(bucket_name, str(shot_number))[1]       
    except swiftclient.ClientException as e: 
        print(f'Pulse did not have {shot_number}!, returning None')
        return None
    
    return pickle.load(BytesIO(my_obj)) 

from typing import List, Dict, Optional
def get_dict_params(shot_number: int, diagnostic_signal_names: List[str], conn: swiftclient.Connection, bucket_name: str = 'JET_PULSES') -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    pulse_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for diagnostic_signal in diagnostic_signal_names: 
        diagnostic, signal = diagnostic_signal.split('/')
        naming_string = f'{shot_number}_{diagnostic}_{signal}'
        local_dict: Dict[str, np.ndarray] = {}
        for dimension in ['data', 'time']:
            object_name = f'{naming_string}_{dimension}.npy'     
            try: 
                my_obj = conn.get_object(bucket_name, object_name)[1]       
            except swiftclient.ClientException as e: 
                print(f'Pulse did not have {object_name}!, returning None')
                return None
            data = np.load(BytesIO(my_obj), allow_pickle=True)
            if diagnostic in ['efit', 'power', 'magn', 'gas']: 
                data = data.item()[dimension]
            local_dict[dimension] = data
        pulse_dict[diagnostic_signal] = local_dict
    return pulse_dict 

def get_mp_names_saved_in_arrays(data_path: str) -> list:
    with open(os.path.join(data_path, 'mp_names_saved.txt'), 'r') as f:
        all_names_str = f.read()
        relevant_mp_columns = all_names_str.split(',')
    return relevant_mp_columns

def get_numpy_arrays_from_local_data(shot_number: int, local_folder_path: 'str'): 
    pulse_string = f'{local_folder_path}/{shot_number}'

    results = []
    for key in ['PROFS', 'MP', 'RADII', 'TIME']: 
        results.append(np.load(f'{pulse_string}_{key}.npy'))
    
    return results, get_mp_names_saved_in_arrays(local_folder_path)

def filter_by_time_window(time_array: np.ndarray, list_return_arrays: List[np.ndarray], t1: float, t2: float) -> List[np.ndarray]:
    bool_window = np.logical_and(time_array > t1, time_array < t2)
    return [arr[bool_window] for arr in list_return_arrays]


from scipy.interpolate import interp1d
REL_COLS = ['BTF', 'IpiFP', 'D_tot', 'PNBI_TOT', 'PICR_TOT','PECR_TOT', 'P_OH', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']
def map_pulse_dict_to_numpy_arrays(pulse_dict: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]], return_torch=False) -> List[np.ndarray]: 
    profile_data, mp_data = pulse_dict['profiles'], pulse_dict['machine_parameters']
    if mp_data.get('PECR_TOT', None) is None: 
        mp_data['PECR_TOT'] = {}
        mp_data['PECR_TOT']['time'], mp_data['PECR_TOT']['data'] = 'NO ECRH USED', None

    ida_times, ne, te, radius = torch.from_numpy(profile_data['time']), torch.from_numpy(profile_data['ne']), torch.from_numpy(profile_data['Te']), torch.from_numpy(profile_data['radius'])
    profiles = torch.stack((ne, te), 1)
    avail_cols = [key for key in REL_COLS if key in mp_data.keys()]
 
    MP_TIME_LIST = torch.Tensor([(min(mp_data[key]['time']), max(mp_data[key]['time'])) for key in avail_cols if isinstance(mp_data[key], dict) and not isinstance(mp_data[key]['time'], str) and mp_data.get(key) is not None])
    
    MP_OBSERVATIONAL_END_TIME, MP_OBSERVATIONAL_START_TIME = MP_TIME_LIST.min(0)[0][1], torch.max(MP_TIME_LIST, 0)[0][0]
    IDA_OBSERVATIONAL_END_TIME, IDA_OBSERVATIONAL_START_TIME = ida_times[-1], ida_times[0]
    
    t1, t2 = max(MP_OBSERVATIONAL_START_TIME, IDA_OBSERVATIONAL_START_TIME), min(MP_OBSERVATIONAL_END_TIME, IDA_OBSERVATIONAL_END_TIME)
    
    relevant_time_windows_bool: torch.Tensor = torch.logical_and(ida_times > t1, ida_times < t2)
    relevant_time_windows: torch.Tensor = ida_times[relevant_time_windows_bool]
    
    relevant_profiles = profiles[relevant_time_windows_bool]
    relevant_radii = radius[relevant_time_windows_bool]

    relevant_machine_parameters: torch.Tensor = torch.empty((len(relevant_profiles), len(REL_COLS)))
    
    for mp_idx, key in enumerate(REL_COLS): 
        relevant_mp_vals = torch.zeros(len(relevant_profiles))
        if not mp_data.get(key): # check for key! 
            mp_raw_data, mp_raw_time = None, None
        else: 
            mp_raw_data, mp_raw_time = mp_data[key]['data'], mp_data[key]['time']
        if mp_raw_time is None or isinstance(mp_raw_time, str): # this catches whenever NBI isn't working or the string in JET pulse files 'NO_ICRH_USED'
            pass 
        else:
            f = interp1d(mp_raw_time, mp_raw_data)
            relevant_mp_vals = torch.from_numpy(f(relevant_time_windows))
        relevant_machine_parameters[:, mp_idx] = relevant_mp_vals
    if return_torch: 
        return relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_time_windows, REL_COLS
    else: 
        return relevant_profiles.numpy(), relevant_machine_parameters.numpy(), relevant_radii.numpy(), relevant_time_windows.numpy(), REL_COLS