from locust import HttpUser, between, task

# call:
# locust -f benchmark.py --headless --host http://0.0.0.0:8000 -u 20 -r 50

class WebsiteUser(HttpUser):
    wait_time = between(2, 4)

    @task
    def attempt(self):
        self.client.get("/app/?sepallength=6.3&sepalwidth=2.5&petallength=4.9&petalwidth=1.5")
