# Group G204

Problem: **MP310** --- Ration Shop Distribution (Dehradun)

Faculty: Saurabh Shrivastava   Batch: 49

Members:
- S24CSEU1915  Prapti Joshi
- S24CSEU1928  SHIVAMJHAKUMAR
- S24CSEU1912  Nisidh kumar
- S24CSEU1900  ANURAG MISHRA
- S24CSEU1927  AARUSHI BRAR

## Docker

Build the API image from the repository root:

```bash
docker build -t mp310-api:1.0 .
```

Run the container and publish its port 8000 on the host:

```bash
docker run --rm -p 8000:8000 --name mp310-api-container mp310-api:1.0
```

Open `http://localhost:8000/docs` for Swagger UI. The image contains the API
and `models/model.joblib`; it does not include training data or require MLflow
at prediction time. The model artifact must be valid for `/health` to return
HTTP 200.

## Kubernetes

The `mp310-api` Deployment runs two replicas of `mp310-api:1.0` on container
port 8000. The `mp310-svc` ClusterIP Service exposes port 80 and forwards to
port 8000. Readiness and liveness probes both use `/health`; liveness has a
longer initial delay so the process has more time to start.

```bash
kubectl apply -f k8s/
kubectl get deployments
kubectl get pods
kubectl get services
kubectl describe deployment mp310-api
kubectl port-forward svc/mp310-svc 8000:80
```

The probes require a loadable model. While `models/model.joblib` is empty,
`/health` returns HTTP 503; Pods will not become Ready, and repeated liveness
probe failures can restart them.
