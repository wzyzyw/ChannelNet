
import socket
 
s=socket.socket()
s.bind(('127.0.0.1',50088))
s.listen()
sync_robot_loc=[12.3,13.6]
def def_socket_thread():
    # global loop
    # loop = asyncio.get_event_loop()
 
    try:
        while True:
            c, addr = s.accept()
            content = read_from_client(c)
            if content.find('location') > -1:
                global sync_robot_loc
                print('receive request')
                print('sycn position=', sync_robot_loc)
                x = sync_robot_loc[0]
                y = sync_robot_loc[1]
                str_content = 'x=' + str(x) + ',y=' + str(y)
                c.send(str_content.encode('ascii'))
                print('finish location send')
            else:
                print('no request')
    except IOError as e:
        print(e.strerror)
    print('start socket thread!!!')
 
 
def read_from_client(c):
    try:
        return c.recv(1024).decode('ascii')
    except IOError as e:
        # 如果异常的话可能就是会话中断 那么直接删除
        print(e.strerror)
 
if __name__=='__main__':
    def_socket_thread()
