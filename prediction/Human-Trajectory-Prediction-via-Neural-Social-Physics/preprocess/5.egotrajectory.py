import os
import numpy as np
import sys
import json
from pathlib import Path
import yaml
import utm
import pdb
import multiprocessing
from tqdm import tqdm
import pypcd

# open json
def read_json(path):    
    with open(path, 'r') as f:
        json_1 = json.load(f)
    return json_1

# save json
def save_json(path, save_dict):
    with open(path, 'w') as f:
        json.dump(save_dict, f)

def time_delta(a, b):
    return np.abs(a-b)

def euler_to_rotmat(euler_angles):
    """
    Convert Euler angles to rotation matrix using X-Y-Z convention
    :param euler_angles: array-like object of Euler angles (in radians)
    :return: 3x3 rotation matrix
    """
    roll, pitch, yaw = euler_angles
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
    rot_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
    rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    rot_mat = np.dot(rot_z, np.dot(rot_y, rot_x))
    return rot_mat

def rotmat_to_euler(rot_mat):
    """
    Convert rotation matrix to Euler angles using X-Y-Z convention
    :param rot_mat: 3x3 rotation matrix
    :return: array-like object of Euler angles (in radians)
    """
    sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to euler angles.

    :param quaternion: numpy array of shape (4,), representing quaternion in wxyz order.
    :return: numpy array of shape (3,), representing euler angles in roll-pitch-yaw order, in radians.
    """
    qw, qx, qy, qz = quaternion
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def euler_to_quaternion(euler):
    """
    Convert euler angles to quaternion.

    :param euler: numpy array of shape (3,), representing euler angles in roll-pitch-yaw order, in radians.
    :return: numpy array of shape (4,), representing quaternion in wxyz order.
    """
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])


def make_rot_mat(string):
    xyzypr = []
    for s in string.split(' '):
        if s != "":
            xyzypr.append(float(s))
    xyz = xyzypr[:3]
    ypr = xyzypr[3:6]
    R = euler_to_rotmat([ypr[2], ypr[1], ypr[0]])
    mat = np.concatenate([R, np.array([xyz]).T], axis=1)
    mat = np.concatenate([mat, np.array([[0,0,0,1]])])
    
    return mat

extrinsic_place={}
extrinsic_place["230409_sungsu_2"] = []
roll,pitch,yaw,X,Y,Z=(-1.5801164867461783, 0.027151080542834397, -0.29717004443411255, 0.0683367584645344, 0.13176968066994704, 0.22080063323529278)
extrinsic_place["230409_sungsu_2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.575738477564237, 0.012918798195376118, 0.952506590602695, -0.11441780226997045, 0.07713486464332951, 0.22688811288419805)
extrinsic_place["230409_sungsu_2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5727644354007912, 0.01462941402073337, 2.2104003425818486, -0.10164049768023917, -0.1073155650024347, 0.22705984207275012)
extrinsic_place["230409_sungsu_2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5824486651955763, -0.003929775066494368, -1.562777913759401, 0.1214916157638479, -0.014423563042019933, 0.23400505061975346)
extrinsic_place["230409_sungsu_2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5626153549109654, -0.02223901888836292, -2.8162846070682974, 0.07062088722856014, -0.1571014621008622, 0.20978674972055983)
extrinsic_place["230409_sungsu_2"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230409_sungsu"] = []
roll,pitch,yaw,X,Y,Z= -1.5775036088975465, 0.05786083933911426, -0.2868389471605804, 0.042691948550066146, 0.11353663922306617, 0.23192759102247304
extrinsic_place["230409_sungsu"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.549181221445747, 0.029987588838949218, 0.9485211875970324, -0.16160999315552527, 0.07183136761273193, 0.23133050564348684
extrinsic_place["230409_sungsu"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.5545995731984756, 0.0010720426596411983, 2.2070174279418606, -0.13974972021668786, -0.10924709731999536, 0.24973226165838627
extrinsic_place["230409_sungsu"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.5939580362321153, -0.0020393011260302525, -1.5620389995649109, 0.09745137983845636, -0.03210511556574866, 0.23217153697476892
extrinsic_place["230409_sungsu"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.5821520581997839, -0.055205551151260204, -2.8078787668628675, 0.03886906954955566, -0.14538290608469762, 0.23404982035646374
extrinsic_place["230409_sungsu"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230409_sungsu_slam"] = []
roll,pitch,yaw,X,Y,Z= -1.5759578627847288, 0.046299893678250374, -0.28746304285810237, 0.06684276166851283, 0.13125549684735113, 0.23286934888221542
extrinsic_place["230409_sungsu_slam"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.5595926151055064, 0.024722370861222764, 0.9475318523995343, -0.13746849512600373, 0.08969105621682437, 0.2299923369196188
extrinsic_place["230409_sungsu_slam"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.5628068393484085, 0.00935946049102705, 2.206270056772176, -0.11595118768061201, -0.09136916396609289, 0.24896886893680542
extrinsic_place["230409_sungsu_slam"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.5824486651955763, -0.00392977506649437, -1.562777913759401, 0.12149161576384791, -0.014423563042019933, 0.2340050506197535
extrinsic_place["230409_sungsu_slam"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.5766803938840563, -0.044901455100005414, -2.8088570964940818, 0.0628105857024876, -0.12765709827207367, 0.2354107754661499
extrinsic_place["230409_sungsu_slam"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230410_3way"]=[]
roll,pitch,yaw,X,Y,Z=(-1.5966732801796644, 0.01525650251224, -0.2949181272561784, 0.10906263961893377, 0.14246703940294864, 0.25080297301723004)
extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.56002613111095, 0.027524439285741, 0.9461071559290232, -0.11021345564255, 0.0950863978987904, 0.2601681575115943)
extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5400246037390792, -0.0490120734051878, 2.208682741749952, -0.11051913530782212, -0.0936266603405737, 0.2626426429561072)
extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5948324569325594, -0.008021411886329141, -1.5603887508687433, 0.12733049300633897, -0.0115806645091294, 0.24437138104832107)
extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.583292258723444, -0.02900054887927589, -2.8161500394386993, 0.039853899821038805, -0.11585506356260328, 0.2540688604830006)
extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])

# extrinsic_place["230410_3way"]=[]
# roll,pitch,yaw,X,Y,Z=(-1.573998557930863, 0.0670466133570567, -0.2877967583494456, 0.037511524941813736, 0.14704683654459005, 0.20968997438352993)
# extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
# roll,pitch,yaw,X,Y,Z=(-1.539351489887983, 0.029707737770573832, 0.9476379659797267, -0.1668387372360703, 0.10558150677020271, 0.21044253683792952)
# extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
# roll,pitch,yaw,X,Y,Z=(-1.5418768418942574, -0.01391013052484425, 2.2060181940038457, -0.127653144110847, -0.1640963307400105, 0.225289709269653)
# extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
# roll,pitch,yaw,X,Y,Z=(-1.6017270043835516, 0.003982264800816124, -1.5632231219839852, 0.0920964050719496, 0.0013431612115951696, 0.20863858565582)
# extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])
# roll,pitch,yaw,X,Y,Z=(-1.5903538944303743, -0.06064350935455305, -2.8085805155090284, 0.033394688277013355, -0.11187617903276169, 0.21030060011470164)
# extrinsic_place["230410_3way"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230410_exit2"]=[]
roll,pitch,yaw,X,Y,Z=(-1.5662906849196259, 0.06893365884046032, -0.3037637593870746, 0.05280103283607366, 0.13715342403781114, 0.21988573077100076)
extrinsic_place["230410_exit2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.54092334274222, 0.038203736132754, 0.945462943418819, -0.12063071768004262, 0.116200241798827, 0.21839218865350807)
extrinsic_place["230410_exit2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5510802265138117, -0.0479635382024061, 2.2052087450275093, -0.10925246388904924, -0.04789390371492798, 0.2249324972111712)
extrinsic_place["230410_exit2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5967270043835515, 0.008982264800816122, -1.5642231219839853, 0.110964050719496, -0.023431612115951697, 0.21863858565582)
extrinsic_place["230410_exit2"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5818964462877974, -0.029569965817519934, -2.820560375142941, 0.0331502434086821, -0.11176020027203414, 0.20914107781822786)
extrinsic_place["230410_exit2"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230412_hanyang_plz"]=[]
roll,pitch,yaw,X,Y,Z=(-1.5686426646726122, 0.02879478663425507, -0.29091189567841963, 0.05993509200483838, 0.11866930107235517, 0.2522386005601282)
extrinsic_place["230412_hanyang_plz"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.573709954943199, 0.01204818981248859, 0.9435488399897654, -0.1444380859993878, 0.07787548539452281, 0.24542403596564827)
extrinsic_place["230412_hanyang_plz"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5792037200902862, 0.018902109240862056, 2.202314889600698, -0.12395342453853525, -0.10330119449321666, 0.26443182198809534)
extrinsic_place["230412_hanyang_plz"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5635737862555323, -0.002030739595228053, -1.5665576336284908, 0.11401280686175774, -0.02721520412864818, 0.2541060749864545)
extrinsic_place["230412_hanyang_plz"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5724565529317815, -0.02640695681508874, -2.8127311352061257, 0.05489667230291188, -0.14023091554269154, 0.25417286196098504)
extrinsic_place["230412_hanyang_plz"].append([roll,pitch,yaw,X,Y,Z])


extrinsic_place["230412_international"]=[]
roll,pitch,yaw,X,Y,Z=(-1.5707084498174, -0.000743215908017004, -0.3051913127371137, 0.05862013371274714, 0.13823027638657137, 0.2099815953841137)
extrinsic_place["230412_international"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.573835854500088, -0.002252009211370698, 0.9464468070169553, -0.13563401960987892, 0.09684242685973177, 0.21318039418746773)
extrinsic_place["230412_international"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5790483614830844, -0.02084491024862666, 2.205217336877669, -0.11462380019590873, -0.08426954910686185, 0.2463110182362)
extrinsic_place["230412_international"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5635924977873756, -0.0022695713060290027, -1.5636580967141114, 0.12312064434487241, -0.0074963673483024506, 0.22188296573930422)
extrinsic_place["230412_international"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.572236085780528, -0.0065008660690548, -2.8198374700627262, 0.06433244787827028, -0.12068299118199903, 0.21197779946397005)
extrinsic_place["230412_international"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230413_ftc"]=[]
roll,pitch,yaw,X,Y,Z=(-1.5464259257367747, -0.00756120294191303, -0.29230573074062444, 0.0851300049804128, 0.14060691473017006, 0.19940168084703572)
extrinsic_place["230413_ftc"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5850219706685402, -0.021688000110013, 0.929699187029708, -0.12886841803440563, 0.06884602652324077, 0.1884636188910964)
extrinsic_place["230413_ftc"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5938013952958117, 0.03448547228913456, 2.206491118847433, -0.11796783858017776, -0.09215944247643819, 0.20861535803363837)
extrinsic_place["230413_ftc"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5426132543557203, -0.006085507103424143, -1.5620768281073825, 0.11981605621349573, -0.015018615483580447, 0.20298166871258985)
extrinsic_place["230413_ftc"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5619278553725147, -0.007833874004346252, -2.8083455225129557, 0.061223460541524034, -0.12830433835449034, 0.20225670103184584)
extrinsic_place["230413_ftc"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230413_hangwonpark"]=[]
roll,pitch,yaw,X,Y,Z=(-1.5778955380814492, -0.020314927480271906, -0.29302433845175657, 0.08178532565881794, 0.14294257373473024, 0.24748559832521283)
extrinsic_place["230413_hangwonpark"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.6040983433106932, -0.02406113172080551, 0.933524494686071, -0.13207020263879515, 0.08136682917291509, 0.23308708675633397)
extrinsic_place["230413_hangwonpark"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5926826753657544, 0.03420335976639748, 2.2058252362971076, -0.11246331897097896, -0.10669970919554371, 0.202496337668308)
extrinsic_place["230413_hangwonpark"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5279038386731423, -0.0017212647711387604, -1.5627339569829757, 0.11632690650179324, -0.012719352424147215, 0.25087500165750276)
extrinsic_place["230413_hangwonpark"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.561373553334711, 0.00749918296429341, -2.808945326598764, 0.057687059940061916, -0.1259635989832681, 0.24878171033149765)
extrinsic_place["230413_hangwonpark"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z= -1.557973859816752, -0.005244012185617961, -0.2868956492824068, 0.06178181437686049, 0.13295532812741026, 0.24714727662671995
extrinsic_place["230413_hangwonpark"].append([roll,pitch,yaw,X,Y,Z])

extrinsic_place["230413_hanyang_womans"]=[]
roll,pitch,yaw,X,Y,Z=(-1.5718727973080617, 0.021618023522841447, -0.29740116865157634, 0.057654655346677734, 0.13234244848636892, 0.23435612734756)
extrinsic_place["230413_hanyang_womans"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5340983433106932, -0.02406113172080551, 0.943524494686071, -0.13207020263879515, 0.08136682917291509, 0.24308708675633398)
extrinsic_place["230413_hanyang_womans"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.566826753657544, 0.02335976639748, 2.2058252362971076, -0.12246331897097897, -0.1166997091955437, 0.240496337668308)
extrinsic_place["230413_hanyang_womans"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.5706956990768903, -0.012689596034959954, -1.5632164768226022, 0.1456465345087537, -0.021954791476212075, 0.2307415265515402)
extrinsic_place["230413_hanyang_womans"].append([roll,pitch,yaw,X,Y,Z])
roll,pitch,yaw,X,Y,Z=(-1.581373553334711, -0.00749918296429341, -2.818945326598764, 0.057687059940061916, -0.1259635989832681, 0.23878171033149764)
extrinsic_place["230413_hanyang_womans"].append([roll,pitch,yaw,X,Y,Z])




out_semantic_matrix = {}
out_semantic_matrix['230410_exit2'] = np.array([[-6.48716404e-01,  7.61030241e-01, -2.95228193e+06],
                                    [-7.61030241e-01, -6.48716404e-01,  2.94661654e+06],
                                    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
out_semantic_matrix['230410_3way'] = np.array([[-5.05028447e-01,  8.63102698e-01, -3.42421277e+06],
                                        [-8.63102698e-01, -5.05028447e-01,  2.38240709e+06],
                                        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
out_semantic_matrix['230409_sungsu_alley'] = np.array([[ 7.44964405e-01, -6.67104216e-01,  2.52903002e+06],
                                        [ 6.67104216e-01,  7.44964405e-01, -3.31545208e+06],
                                        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
out_semantic_matrix['230409_sungsu_cross'] = np.array([[ 7.44964405e-01, -6.67104216e-01,  2.52903002e+06],
                                        [ 6.67104216e-01,  7.44964405e-01, -3.31545208e+06],
                                        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]) 
out_semantic_matrix['230409_sungsu_2'] = np.array([[-9.90413087e-01, -1.38137313e-01,  8.98371825e+05],
                                        [ 1.38137313e-01, -9.90413087e-01,  4.07237117e+06],
                                        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]) 
out_semantic_matrix['2302xx_2exit'] = np.array([[-6.48716404e-01,  7.61030241e-01, -2.95228193e+06],
                                    [-7.61030241e-01, -6.48716404e-01,  2.94661654e+06],
                                    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
out_semantic_matrix['2302xx_3way'] = np.array([[-5.05028447e-01,  8.63102698e-01, -3.42421277e+06],
                                        [-8.63102698e-01, -5.05028447e-01,  2.38240709e+06],
                                        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
out_semantic_matrix['2302xx_sungsu_alley'] = np.array([[ 9.99085592e-01, -4.27548753e-02, -1.49671456e+05],
                                            [ 4.27548753e-02,  9.99085592e-01, -4.16723227e+06],
                                            [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
out_semantic_matrix['2302xx_sungsu_cross'] = np.array([[ 9.99085592e-01, -4.27548753e-02, -1.49671456e+05],
                                            [ 4.27548753e-02,  9.99085592e-01, -4.16723227e+06],
                                            [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]) 

in_semantic_matrix = {}
in_semantic_matrix['230412_hanyang_plz'] = {'2023-04-12-17-15-20':make_rot_mat("6.72021 -10.7602 0.453339     2.6714 -0.0328035  0.0332244"),
                                    '2023-04-12-17-18-24':make_rot_mat("-0.792777  -8.63622  0.309088    2.80992 0.00318242  0.0193545"),
                                    '2023-04-12-17-21-16':make_rot_mat("-4.42124 -6.64816 0.165212    2.61593 -0.0135371 0.00285866"),
                                    '2023-04-12-17-24-19':make_rot_mat("-14.3582   -2.8262 -0.136141    2.75902 0.00586543 0.00871599"),
                                    '2023-04-12-17-26-57':make_rot_mat("-11.9633  -3.72257 -0.108777    2.87192 0.00579089 -0.0053042"),
                                    }
in_semantic_matrix['230413_hanyang_womans'] = {'2023-04-13-11-44-44':make_rot_mat("-17.2599   23.8105 -0.586412    2.54687 -0.0126974 0.00336044")}

in_semantic_matrix['230412_international'] = {'2023-04-12-17-32-46':make_rot_mat("-1.82975 -2.26758 0.106983 0.0993932 0.0123184 -0.025678"),
                                '2023-04-12-17-43-12':make_rot_mat("24.1508   4.80143 -0.877616   0.224937  0.0154687 -0.0179819"),
                                '2023-04-12-17-46-29':make_rot_mat("23.5915   4.42472 -0.851645   0.163296  0.0165385 -0.0241977"),
                                '2023-04-12-17-49-15':make_rot_mat("-1.86791  -2.38461 0.0668298 0.134665 -3.08387  3.12782"),
                                }                                                                                                               
in_semantic_matrix['230413_ftc']  = {'2023-04-13-13-25-19':make_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549"),
                            '2023-04-13-13-38-35':make_rot_mat("0.88389 -0.931077 0.0167018 1.43376 3.14147 3.10547"),
                            '2023-04-13-13-41-29':make_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217"),}
in_semantic_matrix['230413_hangwonpark'] = {'2023-04-13-12-28-14':make_rot_mat("-0.794114  -1.37272 0.0167796 2.58499 3.10378 -3.1307"),
                                '2023-04-13-12-31-45':make_rot_mat("-0.11828 0.260273 0.129865 0.969964  3.13062  3.10571"),
                                '2023-04-13-12-34-14':make_rot_mat("0.286822  0.104614 0.0129571 2.59174  3.0954 3.13644"),
                                '2023-04-13-12-43-35':make_rot_mat("9.79455   0.65552 -0.138923   2.5932 -3.13237  3.12934"),
                                '2023-04-13-12-45-10':make_rot_mat("9.78504    0.72182 -0.0878919  2.59297 -3.13916  3.13277"),
                                '2023-04-13-12-47-23':make_rot_mat("22.9711   3.36325 -0.348537    3.06245 -0.0309605 -0.0317086"),
                                '2023-04-13-12-37-38':make_rot_mat("0 0 0 0 0 0"),
                                }
in_semantic_matrix['2302xx_hanyang_plz'] = {'2023-02-08-16-00-52':np.array([[   0.582112,   -0.813102,  0.00336065,  -0.0880571,],
                                                                    [   0.813105,    0.582117, 0.000862744,    -8.41659,],
                                                                    [-0.00265779,  0.00223035,    0.999994,    0.277955,],
                                                                    [          0,           0,           0,           1,],]),
                                    '2023-02-08-16-03-01':np.array([[  0.623236,  -0.781557, -0.0272954,    -38.914,],
                                                                    [  0.780726,   0.619799,  0.0794707,    7.98531,],
                                                                    [-0.0451932, -0.0708393,   0.996463,  -0.975523,],
                                                                    [         0,          0,          0,          1,]]),
                                    '2023-02-08-16-04-04':np.array([
                                                                    [ 0.581369,   -0.813633, -0.00342145,    -8.40558],
                                                                    [   0.813639,    0.581358,  0.00373209,    -5.09984],
                                                                    [-0.00104746, -0.00495355,    0.999987,   0.0711326],
                                                                    [          0,           0,           0,           1]
                                                                    ], dtype=np.float64),
                                    '2023-02-08-16-17-47':np.array([
                                                                    [  0.988656,  -0.146565, -0.0328228,   -38.1791],
                                                                    [  0.148618,   0.986215,  0.0727562,    7.10626],
                                                                    [ 0.0217068,  -0.076809,   0.996809,   -1.00529],
                                                                    [         0,          0,          0,          1]
                                                                    ], dtype=np.float64),}
in_semantic_matrix['2302xx_hanyang_woman'] = {'2023-02-09-11-49-45':np.array([
                                                                    [0.760918,   -0.648846, -0.00140557,    -22.0841],
                                                                    [  0.648841,    0.760919, -0.00275272,     23.5037],
                                                                    [0.00285562,   0.0011826,    0.999995,   -0.532438],
                                                                    [         0,           0,           0,           1]
                                                                    ], dtype=np.float64),
                                        '2023-02-09-11-54-58':np.array([
                                                                    [0.716925,    -0.697148,   0.00193089,     -23.8572],
                                                                    [0.697151,     0.716923,  -0.00167581,      23.3879],
                                                                    [-0.000216006,   0.00254755,     0.999997,    -0.514491],
                                                                    [          0,            0,            0,            1]
                                                                    ], dtype=np.float64),
                                        '2023-02-09-12-05-15':np.array([
                                                                    [ 0.545297,   -0.838234, -0.00382867,    -19.8822],
                                                                    [ 0.837354,    0.544503,   0.0485239,     20.7266],
                                                                    [-0.0385897,  -0.0296659,    0.998815,   -0.462463],
                                                                    [        0,           0,           0,           1]
                                                                    ], dtype=np.float64),
                                        }
in_semantic_matrix['2302xx_internat'] = {'2023-02-09-09-56-52':np.array([
                                                                [-0.365955,    0.930479,   0.0168913,   -0.501783],
                                                                [-0.929846,   -0.366331,   0.0344745,    0.156359],
                                                                [0.0382656, -0.00309022,    0.999263, -0.00835417],
                                                                        [ 0,           0,           0,           1]
                                                                ], dtype=np.float64),
                                '2023-02-09-09-59-42':np.array([
                                                                [-0.223574,    0.973046,   0.0565293,     19.4792],
                                                                [-0.974668,   -0.222837,  -0.0191102,     1.98601],
                                                                [-0.00599825,  -0.0593698,    0.998218,   -0.779592],
                                                                [          0,           0,           0,           1]
                                                                ], dtype=np.float64),
                                '2023-02-09-10-01-31':np.array([  
                                                                [-0.860453,    0.509288,    0.015732,   -0.816914],
                                                                [-0.508516,   -0.860277,   0.0365266,   -0.162258],
                                                                [0.0321364,   0.0234294,    0.999209, -0.00229016],
                                                                [       0,           0,           0,           1]
                                                                ], dtype=np.float64)
                                }                                     

IL_mat=np.array(
[[0.997242,     -0.00723046,    -0.0738613,     0       ],
 [0.00687393,   0.999963,       -0.00508015,    -0.0077  ],
 [0.0738954,    0.00455843,     0.997256,       -0.07646 ],
 [0,            0,              0,              1       ]])
# [[0.996548,     0.0806576,      -0.0196384,     0       ],
#  [-0.0806414,   0.996742,       0.00161381,     0.0077  ],
#  [0.0197046,    -2.45675e-05,   0.999806,       (0.29 - 0.07646)],#0.07646 ],
#  [0,            0,              0,              1       ]])

l2l_extrinsic_dict = {} # x, y, z, yaw, pitch, roll
l2l_extrinsic_dict['230409_sungsu_2'] = np.array(list(map(float, "-0.0269386  0.0365429  -0.270444 3.13486 -3.1324 3.13908".split( ))))
l2l_extrinsic_dict['230410_exit2'] = np.array(list(map(float, "-0.0159841  0.0560808  -0.233834  3.13606 -3.13419 -3.14082".split( )))) 
l2l_extrinsic_dict['230413_hangwonpark'] = {}
l2l_extrinsic_dict['230413_hangwonpark']['not_slam'] = np.array(list(map(float, "-0.00517077  0.00527534   -0.264051  3.13167 -3.13721  3.13831".split( ))))
l2l_extrinsic_dict['230413_hangwonpark']['slam'] = np.array(list(map(float, "-0.020538 0.0245825 -0.260824  3.13496 -3.13013 -3.13996".split( ))))
l2l_extrinsic_dict['230409_sungsu'] = np.array(list(map(float, "-0.0406384  0.0122286  -0.244534 3.13589 3.13858 3.13822".split( ))))
l2l_extrinsic_dict['230412_hanyang_plz'] = np.array(list(map(float, "0.00322293  0.0129234  -0.265788  3.13582 -3.13063  3.13738".split( )))) 
l2l_extrinsic_dict['230413_hanyang_womans'] = np.array(list(map(float, "-0.000438581    0.0176895    -0.265236  3.13214 -3.12977  3.14141".split( ))))
l2l_extrinsic_dict['230409_sungsu_slam'] = np.array(list(map(float, "-0.0195946  0.0159329  -0.213553 3.13642 3.13843 3.13706".split( ))))
l2l_extrinsic_dict['230412_international'] = np.array(list(map(float, "-0.00393884  0.00228286   -0.258024  3.13426 -3.13227 -3.13772".split( ))))
l2l_extrinsic_dict['230410_3way'] = np.array(list(map(float, "-0.0155146   0.026249  -0.266463  3.13382 -3.13143  3.13922".split( ))))
l2l_extrinsic_dict['230413_ftc'] = np.array(list(map(float, "-0.00907978   0.0248491   -0.222904  3.13511 -3.13645 -3.13855".split( ))))



def euler_to_rotmat2(euler_angles):
    roll, pitch, yaw = euler_angles
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
    rot_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
    rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    rot_mat = np.dot(rot_z, np.dot(rot_y, rot_x))
    return rot_mat

def xyzypr_to_rot_mat(string):
    xyzypr = []
    for s in string.split(' '):
        if s != "":
            xyzypr.append(float(s))
    xyz = xyzypr[:3]
    ypr = xyzypr[3:6]
    R = euler_to_rotmat2([ypr[2], ypr[1], ypr[0]])
    mat = np.concatenate([R, np.array([xyz]).T], axis=1)
    mat = np.concatenate([mat, np.array([[0,0,0,1]])])
    return mat



to_sem_mat = []
# 2023-04-12-17-18-24
old_semantic_mat = xyzypr_to_rot_mat("-0.792777  -8.63622  0.309088    2.80992 0.00318242  0.0193545")
semantic_mat = xyzypr_to_rot_mat("-0.800868  -8.62386  0.272736      1.32639     0.017442 -0.000945303")
old_sem_mat_inv = np.linalg.inv(old_semantic_mat)
to_sem_mat.append(semantic_mat@old_sem_mat_inv)
# 2023-04-12-17-21-16
old_semantic_mat = xyzypr_to_rot_mat("-4.42124 -6.64816 0.165212    2.61593 -0.0135371 0.00285866")
semantic_mat = xyzypr_to_rot_mat("-4.32775 -6.47274 0.178147   0.998814 -0.0080133 -0.0123")
old_sem_mat_inv = np.linalg.inv(old_semantic_mat)
to_sem_mat.append(semantic_mat@old_sem_mat_inv)



ftc_to_sem_mat = []

old_semantic_mat = xyzypr_to_rot_mat("-0.378413  0.501727 0.0871115   0.728172  -0.014466 0.00957549")
semantic_mat = xyzypr_to_rot_mat("-0.335307  0.490096 0.0644958     1.53061  -0.0189423 -0.00241017")
old_sem_mat_inv = np.linalg.inv(old_semantic_mat)
ftc_to_sem_mat.append(semantic_mat@old_sem_mat_inv)

old_semantic_mat = xyzypr_to_rot_mat("0.88389 -0.931077 0.0167018 1.43376 3.14147 3.10547")
semantic_mat = xyzypr_to_rot_mat("0.89041  -0.893881 -0.0107157  0.122374 0.0318579 0.0123532")
old_sem_mat_inv = np.linalg.inv(old_semantic_mat)
ftc_to_sem_mat.append(semantic_mat@old_sem_mat_inv)


old_semantic_mat = xyzypr_to_rot_mat("1.5422  0.451621 0.0189186    2.03641 -0.0480602 0.00366217")
semantic_mat = xyzypr_to_rot_mat("0.305223   2.4823 0.192273    1.54488 -0.0184657  0.0828688")
old_sem_mat_inv = np.linalg.inv(old_semantic_mat)
ftc_to_sem_mat.append(semantic_mat@old_sem_mat_inv)

# imu_calib =\
# "-0.998557 0.0359814 0.039867 0.0348208 0.99896 -0.0294344 -0.0408847 -0.0280037 -0.998771"
# imu_calib = list(map(float, imu_calib.split(' ')))
# imu_lidar_extrinsic[0,:3] = imu_calib[:3]
# imu_lidar_extrinsic[1,:3] = imu_calib[3:6]
# imu_lidar_extrinsic[2,:3] = imu_calib[6:9]
# print(imu_lidar_extrinsic)
inverse = True


def process(img_list, yaml_path, tf_list, scene_path, place, scene,\
            imu_list, rtk_list, l2l_extrinsic, extrinsic, loc_save_path):
    for id, frame in enumerate(img_list):

        for name in [1, 2, 3, 4, 5]:
            if len(tf_list) != 0: #in_door preprocessing
                t_delta = time_delta(np.array(frame), tf_list)
                idx = np.where(t_delta == np.min(t_delta))

                pose = open(str(scene_path) + '/tf/odometry/{}.txt'.format(tf_list[idx[0].item()]), 'r').read()
                LG_pose_list = [float(i) for i in pose.split(" ")]
                indoor_flag = True

                LG_quaternion = np.array(LG_pose_list[6:]+LG_pose_list[3:6]) # w, x, y, z
                LG_rpy = quaternion_to_euler(LG_quaternion)
                
                # ego motion : point cloud compensation
                LG_R = euler_to_rotmat(LG_rpy) #ego rot
                LG_T = np.array([[LG_pose_list[0], LG_pose_list[1], LG_pose_list[2]]])
                LG_mat = np.concatenate([LG_R, LG_T.T], axis=1)
                LG_mat = np.concatenate([LG_mat, np.array([[0,0,0,1]])])

                indoor_key_list = in_semantic_matrix[place].keys()
                mul_semantic = False
                for key in indoor_key_list:
                    if key in scene:
                        semantic_matrix = in_semantic_matrix[place][key]
                        
                        cur_to_sem_mat = None
                        if '12-17-18-24_sync_odom' in str(scene_path) and ('cut1' in str(scene_path) or 'cut4' in str(scene_path)):
                            cur_to_sem_mat = to_sem_mat[0]
                        elif '12-17-18-24_sync_odom' in str(scene_path) and not ('cut1' in str(scene_path) or 'cut4' in str(scene_path)):
                            cur_to_sem_mat = to_sem_mat[1]
                        if 'indoor_1_2023-04-13-13-25-19_sync_odom_sync_offset-2_-2_-2_-1_-1_cut3_20sec' in str(scene_path):
                            cur_to_sem_mat = ftc_to_sem_mat[0]
                        elif 'indoor_1_2023-04-13-13-25-19_sync_odom_sync_offset-2_-2_-2_-1_-1_cut7_20sec' in str(scene_path):
                            cur_to_sem_mat = ftc_to_sem_mat[0]
                        elif 'indoor_1_2023-04-13-13-41-29_sync_odom_sync_offset0_0_0_1_0_cut2_20sec' in str(scene_path):
                            cur_to_sem_mat = ftc_to_sem_mat[2]
                        if str(type(cur_to_sem_mat)) == "<class 'numpy.ndarray'>":
                            semantic_matrix = cur_to_sem_mat@semantic_matrix # v1

                        LG_mat = semantic_matrix@LG_mat
                        mul_semantic = True
                        break
                assert mul_semantic == True

                file = open(str(scene_path) + "/ego_trajectory/{}.txt".format(frame), "w")
                with open(str(scene_path) + "/ego_trajectory/{}.txt".format(frame), "w") as file:
                    file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(LG_mat[0][0], LG_mat[0][1], LG_mat[0][2], LG_mat[0][3],\
                                                                                            LG_mat[1][0], LG_mat[1][1], LG_mat[1][2], LG_mat[1][3],\
                                                                                            LG_mat[2][0], LG_mat[2][1], LG_mat[2][2], LG_mat[2][3],\
                                                                                            LG_mat[3][0], LG_mat[3][1], LG_mat[3][2], LG_mat[3][3],
                                                                                        ))
                # file = open(loc_save_path + f"odom/{place}/{scene}/" + "{}.txt".format(frame), "w")
                # with open(loc_save_path + f"odom/{place}/{scene}/" + "{}.txt".format(frame), "w") as file:
                #     file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(LG_mat[0][0], LG_mat[0][1], LG_mat[0][2], LG_mat[0][3],\
                #                                                                             LG_mat[1][0], LG_mat[1][1], LG_mat[1][2], LG_mat[1][3],\
                #                                                                             LG_mat[2][0], LG_mat[2][1], LG_mat[2][2], LG_mat[2][3],\
                #                                                                             LG_mat[3][0], LG_mat[3][1], LG_mat[3][2], LG_mat[3][3],
                #                                                                             ))

            elif len(imu_list) != 0 and len(rtk_list) != 0: #out_door processing
                pose_quaternion = open(str(scene_path) + '/imu/{}.txt'.format(frame), 'r').read()
                pose_localization = open(str(scene_path) + '/rtk/{}.txt'.format(frame), 'r').read()
                pose_quaternion = list(map(float, pose_quaternion.split(' ')))
                pose_localization = np.array(list(map(float, pose_localization.split(' '))))
                
                IG_mat = np.eye(4)
                # import pdb;pdb.set_trace()
                imu_rpy = quaternion_to_euler([pose_quaternion[3], pose_quaternion[0], pose_quaternion[1], pose_quaternion[2]])
                xy = utm.from_latlon(pose_localization[0], pose_localization[1])
                pose_localization[0], pose_localization[1] = xy[0], xy[1]

                if id == 0: #for init position about gps
                    init_pose = np.array(pose_localization, np.float128)
                    init_rpy = np.array(imu_rpy)
                    init_rot_mat = euler_to_rotmat(init_rpy)
                #     file = open(loc_save_path.split("odom")[0] + "init_{}*{}.txt".format(place, scene), "w")
                #     with open(loc_save_path.split("odom")[0] + "init_{}*{}.txt".format(place, scene), "w") as file:
                #         file.write("{},{},{},{},{},{}".format(pose_localization[0], pose_localization[1], pose_localization[2], imu_rpy[0], imu_rpy[1], imu_rpy[2]))
                    
                pose_localization -= init_pose
                pose_localization = np.linalg.inv(init_rot_mat)@pose_localization
                imu_rpy -= init_rpy
                
                B2T_mat = euler_to_rotmat([l2l_extrinsic[5], l2l_extrinsic[4], l2l_extrinsic[3]])
                B2T_mat = np.concatenate([B2T_mat, np.array([l2l_extrinsic[:3]]).T], axis=1)
                B2T_mat = np.concatenate([B2T_mat, np.array([[0,0,0,1]])])

                IG_mat[:3, :3] = euler_to_rotmat(imu_rpy)
                IG_mat[:3, 3] = np.array(pose_localization)
                if inverse:
                    LG_mat = IG_mat@B2T_mat@np.linalg.inv(IL_mat)
                else:
                    LG_mat = IG_mat@B2T_mat@IL_mat
                indoor_flag=False

                file = open(str(scene_path) + "/ego_trajectory/{}.txt".format(frame), "w")
                with open(str(scene_path) + "/ego_trajectory/{}.txt".format(frame), "w") as file:
                    file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(LG_mat[0][0], LG_mat[0][1], LG_mat[0][2], LG_mat[0][3],\
                                                                                            LG_mat[1][0], LG_mat[1][1], LG_mat[1][2], LG_mat[1][3],\
                                                                                            LG_mat[2][0], LG_mat[2][1], LG_mat[2][2], LG_mat[2][3],\
                                                                                            LG_mat[3][0], LG_mat[3][1], LG_mat[3][2], LG_mat[3][3],
                                                                                            ))
                # file = open(loc_save_path + f"odom/{place}/{scene}/" + "{}.txt".format(frame), "w")
                # with open(loc_save_path + f"odom/{place}/{scene}/" + "{}.txt".format(frame), "w") as file:
                #     file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(LG_mat[0][0], LG_mat[0][1], LG_mat[0][2], LG_mat[0][3],\
                #                                                                             LG_mat[1][0], LG_mat[1][1], LG_mat[1][2], LG_mat[1][3],\
                #                                                                             LG_mat[2][0], LG_mat[2][1], LG_mat[2][2], LG_mat[2][3],\
                #                                                                             LG_mat[3][0], LG_mat[3][1], LG_mat[3][2], LG_mat[3][3],
                #                                                                             ))

    


def main():
    loc_save_path = '/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/first_loc/'
    os.makedirs(loc_save_path, exist_ok=True)

    # data_path = "/mnt/sdc/SiTDataset_s20_sdc/data_preprocess/8.convert_to_pcd_img/" #rosbag -> raw data 변환 한거
    # data_path = "/mnt/sdc/SiTDataset_s20_sdc/data_preprocess/7.make_to_5hz/"
    data_path = "/mnt/sdc/jhkim20/deepen_anno_3d_projection/convert_to_pcd_img"

    import pandas as pd
    csv_file = pd.read_csv('/media/hdd/jyyun/husky/Human-Trajectory-Prediction-via-Neural-Social-Physics/preprocess/Scene_cutting_.csv')
    csv_file = csv_file.fillna("_")
    csv_file.iloc[0]
    csv_list = []
    for i in range(len(csv_file)):
        date = csv_file.iloc[i].Date
        Location = csv_file.iloc[i].Location
        File_name = csv_file.iloc[i].File_name
        Duration = csv_file.iloc[i].Duration
        selected = csv_file.iloc[i].Selected
        modified_name = csv_file.iloc[i].Modified_name
        Cut_id = csv_file.iloc[i].cut_id
        time_offset = csv_file.iloc[i]['Offset Crosscheck']
        if selected == "*" or selected == "o":
            try:
                csv_list.append([File_name, time_offset, modified_name])
            except:
                import pdb;pdb.set_trace()
                


    place_list = os.listdir(data_path)
    place_list.sort()
    n_jobs = 30
    p = multiprocessing.Pool(n_jobs)
    res = []
    
    for place in place_list:
        place_path = Path(data_path, place)
        if place in ["ImageSets"]:
            continue
        scene_list = os.listdir(place_path)
        if place in ["token_list"]:
            continue
        for scene in scene_list:
            # print(scene)
            # break
            scene_path = Path(place_path, scene)

            os.makedirs(str(scene_path) + "/ego_trajectory/", exist_ok=True) # for detection inverse compensation
            os.makedirs(str(scene_path) + '/tf/odometry/', exist_ok=True)
            
            tf_list = os.listdir(str(scene_path) + '/tf/odometry/') #lio-sam data
            tf_list = [int(i.split(".txt")[0]) for i in tf_list]
            tf_list.sort()
            tf_list = np.array(tf_list)

            imu_list = os.listdir(str(scene_path) + '/imu/') 
            imu_list = [int(i.split(".txt")[0]) for i in imu_list]
            imu_list.sort()
            imu_list = np.array(imu_list)

            rtk_list = os.listdir(str(scene_path) + '/rtk/') #position
            rtk_list = [int(i.split(".txt")[0]) for i in rtk_list]
            rtk_list.sort()
            rtk_list = np.array(imu_list)
            
            place_check = str(place)
            # if False:
            #     pass
            # elif "way" in place_check:
            #     pass
            # elif "exit" in place_check:
            #     pass
            # elif "ftc" in place_check:
            #     pass
            # elif "inter" in place_check:
            #     pass
            # elif "woman" in place_check:
            #     pass
            # elif "hangwon" in place_check:
            #     pass
            # elif "sungsu" in place_check:
            #     pass
            # elif "sungsu_slam" in place_check:
            #     pass
            # elif "sungsu2" in place_check:
            #     pass
            # else:
            #     continue
            # if "indoor_1_2023-04-13-13-38-35_sync_odom_sync_offset-2_-2_-2_-1_-2_cut3_20sec_5hz" not in scene:
            #     continue
            
            if "230413_hangwonpark" == place:
                if "slam" in scene:
                    l2l_extrinsic = l2l_extrinsic_dict[place]['not_slam']
                else:
                    l2l_extrinsic = l2l_extrinsic_dict[place]['slam']
            else:
                l2l_extrinsic = l2l_extrinsic_dict[place]
            yaml_path = '/mnt/sdc/SiTDataset_s20_sdc/data_preprocess/csv/intrinisc_for_0513/'
            extrinsic = extrinsic_place[place]
            
            rgb_img_path = Path(scene_path, "cam_img")
            save_path = rgb_img_path
        
            img_list = os.listdir(str(rgb_img_path) + "/" + str(1) + "/data/")
            img_list = [int(i.split(".png")[0]) for i in img_list]
            img_list.sort()
            # os.makedirs(loc_save_path+f"odom/{place}/{scene}", exist_ok=True)
            print(scene)
            for i in range(len(csv_list)):
                if csv_list[i][0] not in str(scene_path):
                    continue
                if len(img_list) == 0 or len(img_list) == 95:
                    continue
                print(csv_list[i][2])
                print(place, scene)
                process(img_list, yaml_path, tf_list, scene_path, place, scene, imu_list, rtk_list, l2l_extrinsic, extrinsic, loc_save_path)
                # res.append(p.apply_async(process, 
                #             kwds=dict(
                #             img_list=img_list,
                #             yaml_path=yaml_path,
                #             tf_list=tf_list,
                #             scene_path=scene_path,
                #             place=place,
                #             scene=scene,
                #             imu_list=imu_list,
                #             rtk_list=rtk_list,
                #             l2l_extrinsic=l2l_extrinsic,
                #             extrinsic=extrinsic,
                #             loc_save_path=loc_save_path,
                #             )))
                break
            # res.append(p.apply_async(process, 
            #             kwds=dict(
            #              img_list=img_list,
            #              yaml_path=yaml_path,
            #              tf_list=tf_list,
            #              scene_path=scene_path,
            #              place=place,
            #              scene=scene,
            #              imu_list=imu_list,
            #              rtk_list=rtk_list,
            #              l2l_extrinsic=l2l_extrinsic,
            #              extrinsic=extrinsic,
            #              loc_save_path=loc_save_path+f"odom/{place}/{scene}/",
            #              )))
            # process(img_list, yaml_path, tf_list, scene_path, place, scene, imu_list, rtk_list, l2l_extrinsic, extrinsic, loc_save_path)

            # for idx, frame in enumerate(img_list):
            #     print(idx)
                # print("\r{} / {}".format(idx, len(img_list)), end = '')
                
    for r in tqdm(res):
        r.get()

if __name__ == '__main__':
    main()
