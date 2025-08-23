from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def recursive_model_construct(model_class: Type[T], data: dict[str, Any]) -> T:
    """Recursively constructs a Pydantic model and its nested models without running validation."""
    field_types = model_class.model_fields

    constructed_data = {}
    for field_name, value in data.items():
        if field_name in field_types:
            field_type = field_types[field_name].annotation

            # Check if the field type is a Pydantic model
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                constructed_data[field_name] = recursive_model_construct(
                    field_type, value
                )
            # Handle lists of Pydantic models
            elif (
                hasattr(field_type, "__origin__")
                and field_type.__origin__ is list
                and len(field_type.__args__) > 0
                and isinstance(field_type.__args__[0], type)
                and issubclass(field_type.__args__[0], BaseModel)
            ):
                model_in_list = field_type.__args__[0]
                constructed_data[field_name] = [
                    recursive_model_construct(model_in_list, item) for item in value
                ]
            else:
                constructed_data[field_name] = value

    return model_class.model_construct(**constructed_data)
