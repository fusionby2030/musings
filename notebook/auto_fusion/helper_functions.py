from typing import Dict, List
import swiftclient
from io import BytesIO
import numpy as np 
import os 
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
                # print(f'Pulse did not have {object_name}!, returning None')
                return None
            data = np.load(BytesIO(my_obj), allow_pickle=True)
            if diagnostic in ['efit', 'power', 'magn', 'gas']: 
                data = data.item()[dimension]
            local_dict[dimension] = data
        pulse_dict[diagnostic_signal] = local_dict
    return pulse_dict 