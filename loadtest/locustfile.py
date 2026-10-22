"""Load test for POST /recommend.

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 64 -r 16 -t 2m --csv docs/locust

Each simulated user asks for the top 10 of a random known user, with the
default 100 candidates and the fitted beta, which is the request the service
exists to answer. N_USERS bounds the user ids (set it to the export's
n_users); about 2% of requests use an unknown id to exercise the fallback.
"""
import os
import random

from locust import FastHttpUser, constant, task

N_USERS = int(os.environ.get("N_USERS", "90000"))


class Client(FastHttpUser):
    wait_time = constant(0)

    @task
    def recommend(self):
        user = (random.randrange(N_USERS) if random.random() > 0.02
                else N_USERS + random.randrange(1000))
        self.client.post("/recommend", json={"user_id": user, "k": 10},
                         name="/recommend")
