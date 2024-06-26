"""입력받은 task를 별도의 thread에서 차례대로 수행하는 일꾼"""
import queue
import threading
from log_manager import LogManager


class Worker:
    """
    입력받은 task를 별도의 thread에서 차례대로 수행하는 일꾼

    task가 추가되면 차례대로 task를 수행하며, task가 모두 수행되면 새로운 task가 추가 될때까지 대기한다.
    task는 dictionary이며 runnable에는 실행 가능한 객체를 담고 있어야 하며, runnable의 인자로 task를 넘겨준다.
    -> post_task 함수를 통해 queue에 처리해야할 일을 밀어 넣고 해 당 task를 순차적으로 꺼내 실행함 
    Operator는 모듈들을 적절히 사용해서 자동거래를 처리하는 역할을 한다. 
    주기적으로 자동거래만 처리하는 것이 아니라 자동거래를 정지하거나 수익률을 요청하여 반환하는 일도 하기 때문에 Operator는 worker가 필요하다 
    자동거래가 처리되는 중에 다른 요청을 받아서 처리할 수 있어야 하기 때문이다. 
    
    Operator의 Worker 역시 요청받은 일을 순차적으로 처리하고 결과를 callback을 통해서 반환하는 일만을 담당한다. 
    Worker는 자신이 어떤 일을 하는지 알지 못한다  
    """

    def __init__(self, name):
        self.task_queue = queue.Queue() # 요청받은 일을 순차적으로 처리하기 위해 queue를 사용 
        self.thread = None
        self.name = name
        self.logger = LogManager.get_logger(name)
        self.on_terminated = None

    def register_on_terminated(self, callback):
        """종료 콜백 등록"""
        self.on_terminated = callback

    def post_task(self, task):

        """
        task를 queue에 추가해주는 함수 
            -> operator의 def start 함수 내에서 
              self.worker.start와 함께 실행 됨 
              self.worker.post_task({"runnable": self._execute_trading})

        task: dictionary이며 runnable에는 실행 가능한 객체를 담고 있어야 하며, runnable의 인자로 task를 넘겨준다.
        -> simulation에서는 사용하지 않음 -> del task로 제거 해줌 
        """
        self.task_queue.put(task)

    def start(self):
        """
        작업을 수행할 스레드를 만들고 start한다.

        이미 작업이 진행되고 있는 경우 아무런 일도 일어나지 않는다.
        """

        if self.thread is not None: # 작업이 진행 중인 경우 아무것도 안함
            return

        def looper(): # post_task에서 queue로 쌓아놓은 task를 하나씩 꺼내 실행하는 함수 
            while True:
                # 요청받은 일이 없는 상황에도 반복문이 계속 수행되는 것을 막기 위해 queue 객체를 사용함 
                self.logger.debug(f"Worker[{self.name}:{threading.get_ident()}] WAIT ==========")
                task = self.task_queue.get() # queue에 저장된 데이터가 있으면 꺼내주고 
                if task is None:  # 데이터를 모두 꺼내 비어 있는 경우 동작을 멈추고 기다림 (= task_queue.task_done())                                 
                    self.logger.debug(
                        f"Worker[{self.name}:{threading.get_ident()}] Termanited .........."
                        )
                    if self.on_terminated is not None:# 새로운 요청이 None이면 on_terminated 콜백 함수를 이용해 반복문을 종료
                        self.on_terminated()          # -> 반복문이 종료되면 스레드 실행도 종료되고 start 이전 상태로 돌아감
                    
                    self.task_queue.task_done()
                    break
                
                self.logger.debug(f"Worker[{self.name}:{threading.get_ident()}] GO ----------")
                runnable = task["runnable"]
                runnable(task)
                self.task_queue.task_done() # .task_done() is used to mark .join() that the processing is done.
                                            # Queue.task_done lets workers say when a task is done. 
                                            # Someone waiting for all the work to be done with Queue.join will wait 
                                            # until enough task_done calls have been made, not when the queue is empty.
        # print(self.name)
        self.thread = threading.Thread(target=looper, name=self.name, daemon=True) # thread 초기화 
        # daemon=True 여러 작업이 종속적으로 끝날지 독립적으로 끝낼지 정함, True는 스레드 중 하나의 작업이 끝나면 모든 작업이 끝나도록함
        self.thread.start() # 스레드 시작선언

    def stop(self):
        """현재 진행 중인 작업을 끝으로 스레드를 종료하도록 한다."""
        if self.thread is None:
            return

        self.task_queue.put(None)
        self.thread = None
        self.task_queue.join() # queue에 있는 모든 작업이 완료되기까지 기다리는 메서드로 
                               # 큐에 들어간 데이터의 개수만큼 task_queue.task_done()이 
                               # 호출 될 때까지 종료되지 않고 기다릴 수 있게 해줌 
                               # 즉, 요청된 모든 작업이 완료되기까지 stop 메서드가 끝나지 않고 기다림 
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            
                                                            