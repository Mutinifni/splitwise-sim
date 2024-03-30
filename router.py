import logging

from abc import ABC, abstractmethod

from simulator import clock, schedule_event, cancel_event, reschedule_event


class Router(ABC):
    """
    Router routes Requests to Application Schedulers.
    """
    def __init__(self,
                 cluster,
                 overheads):
        self.cluster = cluster
        self.overheads = overheads
        self.applications = []
        self.schedulers = {}

        # request queues
        self.pending_queue = []
        self.executing_queue = []
        self.completed_queue = []

    def add_application(self, application):
        self.applications.append(application)
        self.schedulers[application.application_id] = application.scheduler

    def run(self):
        pass

    @abstractmethod
    def route(self, *args):
        """
        Main routing logic
        """
        raise NotImplementedError

    def request_arrival(self, request):
        request.arrive_at_router()
        self.pending_queue.append(request)
        self.route_request(request)

    def request_completion(self, request):
        request.complete_at_router()
        self.executing_queue.remove(request)
        self.completed_queue.append(request)

    def route_request(self, request):
        self.route(request)
        self.pending_queue.remove(request)
        self.executing_queue.append(request)

    def save_results(self):
        #results = []
        #for request in self.completed_queue:
        #    times = request.time_per_instance_type()
        #    results.append(times)
        #utils.save_dict_as_csv(results, "router.csv")
        pass


class NoOpRouter(Router):
    """
    Forwards request to the appropriate scheduler without any overheads.
    """
    def route(self, request):
        scheduler = self.schedulers[request.application_id]
        f = lambda scheduler=scheduler,request=request: \
            scheduler.request_arrival(request)
        schedule_event(self.overheads.routing_delay, f)
