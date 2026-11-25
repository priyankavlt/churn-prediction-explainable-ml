"""Request / response models. Literal types give free input validation and OpenAPI docs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
NoPhone = Literal["Yes", "No", "No phone service"]
NoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: NoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, le=500)
    TotalCharges: float | None = Field(
        default=None, ge=0, description="Defaults to tenure x MonthlyCharges if omitted"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.7,
                "TotalCharges": 280.5,
            }
        }
    )

    @model_validator(mode="after")
    def _consistent_services(self):
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'")
        internet_addons = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]
        no_internet = self.InternetService == "No"
        if no_internet != all(v == "No internet service" for v in internet_addons):
            raise ValueError(
                "Internet add-ons must be 'No internet service' iff InternetService is 'No'"
            )
        if self.TotalCharges is None:
            self.TotalCharges = round(self.tenure * self.MonthlyCharges, 2)
        return self


class PredictionOut(BaseModel):
    churn_probability: float
    risk_tier: Literal["low", "medium", "high"]
    expected_value_of_contact: float = Field(
        description="Expected $ profit of sending a retention offer to this customer"
    )
    recommend_contact: bool


class ContributionOut(BaseModel):
    feature: str
    value: str | float | int
    shap_value: float = Field(description="Contribution to churn log-odds (positive = riskier)")


class ExplanationOut(PredictionOut):
    base_value: float = Field(description="Model's average log-odds before any feature is seen")
    contributions: list[ContributionOut]
    actions: list[str]


class BatchIn(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=10_000)


class BatchOut(BaseModel):
    predictions: list[PredictionOut]


class ModelInfo(BaseModel):
    model_version: str
    trained_at: str
    data_sha256: str
    threshold: float
    params: dict
    metrics: dict
    economics: dict
    global_importance: dict[str, float]
