import os 

source = [
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-hopper_vel-ACD-20230621-192407',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-hopper_vel-ABCD-20230620-213259',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-hopper_vel-BD-20230620-003414',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-hopper_vel-BD-20230621-124108',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-fetch-ABCD-20230622-212210',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-fetch-ABCD-20230623-143515',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-swimmer_dir-ACD-20230621-192411',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-swimmer_dir-ABCD-20230622-192221',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-swimmer_dir-ABCD-20230614-220749',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-swimmer_dir-ABCD-20230624-095727',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-ur5e-BD-20230621-124115',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-ur5e-ABCD-20230622-192216',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-ur5e-ACD-20230621-192342',
    '/data/floyed/meta/train/PGPE-MetaStdpMLPPolicy-ur5e-BD-20230620-003358',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-ant_dir-ABCD-20230614-000507',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-ant_dir--20230622-113608',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-halfcheetah_vel-ABCD-20230614-000505',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-halfcheetah_vel--20230622-113532',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-hopper_vel--20230222-142648',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-hopper_vel--20230622-113649',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-fetch--20230622-132230',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-fetch--20230623-143537',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-fetch--20230623-143555',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-swimmer_dir--20230224-112334',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-swimmer_dir--20230622-113637',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-ur5e--20230410-001319',
    '/data/floyed/meta/train/PGPE-MLPSnnPolicy-ur5e--20230622-113621',
]

_type = [
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    'ABCD',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
]

for idx, dir in enumerate(source):
    path = dir.split('/')[-1]
    tt = path.split('-')[-3]
    if tt != _type[idx] and 'MLPSnn' not in path:
        path = path.replace(tt, _type[idx])
        print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    os.system('scp -r floyed@210.75.240.144:{}/log.txt {}'.format(dir, path))
    os.system('scp -r floyed@210.75.240.144:{}/summary.csv {}'.format(dir, path))
    
    if tt != _type[idx] and 'MLPSnn' not in path:
        with open(os.path.join(path, 'log.txt'), 'r') as file:
            content = file.read()

            # 替换字符串
            content = content.replace(tt, _type[idx])

            # 将修改后的内容写回文件
            with open(os.path.join(path, 'log.txt'), 'w') as file:
                file.write(content)