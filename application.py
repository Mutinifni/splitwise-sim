import logging

from abc import ABC, abstractmethod

import model_repo
import orchestrator_repo

from metrics import ApplicationMetrics, ApplicationSLO
from simulator import clock, schedule_event, cancel_event, reschedule_event


class Application():
    """
    An Application is the endpoint that a Request targets.
    Applications can have any number of Instances which all serve the same model.
    Requests are scheduled to Instances by the Scheduler.
    Application Instances can be autoscaled by the Allocator.
    """
    def __init__(self,
                 application_id,
                 model_architecture,
                 model_size,
                 cluster,
                 router,
                 arbiter,
                 overheads,
                 scheduler=None,
                 allocator=None,
                 instances=None):
        self.application_id = application_id

        # hardware
        self.processors = []

        # model
        self.model_architecture = model_architecture
        self.model_size = model_size

        # orchestration
        if instances is None:
            self.instances = []
        self.cluster = cluster
        self.scheduler = scheduler
        self.allocator = allocator
        self.router = router
        self.arbiter = arbiter

        # overheads
        self.overheads = overheads

        # metrics
        self.metrics = ApplicationMetrics()
        self.slo = ApplicationSLO()

    def add_instance(self, instance):
        """
        Application-specific method to add an instance to the application.
        """
        self.instances.append(instance)
        self.scheduler.add_instance(instance)

    def get_results(self):
        allocator_results = self.allocator.get_results()
        scheduler_results = self.scheduler.get_results()
        self.scheduler.save_all_request_metrics()
        return allocator_results, scheduler_results

    @classmethod
    def from_config(cls, *args, cluster, arbiter, router, **kwargs):
        # parse args
        application_cfg = args[0]

        # get model
        model_architecture_name = application_cfg.model_architecture
        model_size_name = application_cfg.model_size
        model_architecture = model_repo.get_model_architecture(model_architecture_name)
        model_size = model_repo.get_model_size(model_size_name)

        # get orchestrators
        allocator_name = application_cfg.allocator
        scheduler_name = application_cfg.scheduler
        application = cls(application_id=application_cfg.application_id,
                          model_architecture=model_architecture,
                          model_size=model_size,
                          cluster=cluster,
                          router=router,
                          arbiter=arbiter,
                          overheads=application_cfg.overheads)
        allocator = orchestrator_repo.get_allocator(allocator_name,
                                                    arbiter=arbiter,
                                                    application=application,
                                                    debug=application_cfg.debug)
        scheduler = orchestrator_repo.get_scheduler(scheduler_name,
                                                    router=router,
                                                    application=application,
                                                    debug=application_cfg.debug)
        application.scheduler = scheduler
        application.allocator = allocator
        return application
